"""
DeltaNet: Evolved Linear Attention Architecture with Multi-Timescale Gated Retention
This evolution integrates:
- Multi-timescale (K) gated decays for associative memory
- Normalized readout with running denominator to mitigate drift
- Parallel local causal attention with ALiBi bias for short-range selectivity
- Chunkwise streaming computation for efficiency and sub-quadratic complexity
- einops.rearrange for all reshaping operations

Maintains compatibility with existing model/block interfaces and preserves
sub-quadratic complexity while improving expressiveness and long-context handling.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _build_alibi_slopes(num_heads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create ALiBi slopes for each head.
    Follows standard approach: larger slope for lower-indexed heads.
    Returns tensor of shape [num_heads].
    """
    # Implementation adapted from ALiBi reference
    def get_slopes(n):
        # This function is from the ALiBi paper implementation
        def get_power_of_two_slopes(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_power_of_two_slopes(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_power_of_two_slopes(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(get_slopes(num_heads), device=device, dtype=dtype)
    return slopes


class DeltaNet(nn.Module):
    """Single DeltaNet attention layer with multi-timescale gated retention and local attention.

    forward(x: Tensor, mask: Optional[Tensor] = None) -> Tensor
    Input:  x [batch, seq_len, hidden_size]
            mask [batch, seq_len] with 1 for valid tokens, 0 for padding (optional)
    Output: y [batch, seq_len, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        timescales: int = 2,
        window_size: int = 64,
        chunk_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.timescales = timescales
        self.window_size = window_size
        self.chunk_size = chunk_size

        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        )

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Gating for memory update (per head per timescale)
        self.gate_proj = nn.Linear(hidden_size, num_heads * timescales, bias=True)
        # Gate to mix local vs linear outputs (per head)
        self.alpha_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Learned per-head per-timescale decay logits -> sigmoid to (0,1)
        self.decay_logit = nn.Parameter(torch.zeros(num_heads, timescales))

        # Normalization epsilon
        self.eps = 1e-6

        # Stabilization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Initialize gate biases: moderate update and balanced mix
        nn.init.constant_(self.gate_proj.bias, -0.1)
        nn.init.constant_(self.alpha_proj.bias, 0.0)
        # Initialize decays for multiple timescales: one near long, one medium
        with torch.no_grad():
            if self.timescales >= 1:
                long_decay = torch.logit(torch.tensor(0.99)).item()
                self.decay_logit[:, 0].fill_(long_decay)
            for k in range(1, self.timescales):
                p = 0.8 ** (k)
                med_decay = torch.logit(torch.tensor(p)).item()
                self.decay_logit[:, k].fill_(med_decay)

    @torch.compile(mode="default", fullgraph=False)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore
        b, t, d_model = x.shape
        h, d = self.num_heads, self.head_dim
        k_times = self.timescales

        residual = x

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [b, t, h, d]
        q = rearrange(q, "b t (h d) -> b t h d", h=h)
        k = rearrange(k, "b t (h d) -> b t h d", h=h)
        v = rearrange(v, "b t (h d) -> b t h d", h=h)

        # Gates
        gate_raw = self.gate_proj(x)  # [b, t, h*k]
        gate = torch.sigmoid(rearrange(gate_raw, "b t (h k) -> b t h k", h=h, k=k_times))  # [b,t,h,k]
        alpha = torch.sigmoid(rearrange(self.alpha_proj(x), "b t h -> b t h"))  # [b,t,h]

        # Optional mask handling (1 for valid, 0 for pad)
        if mask is not None:
            # Ensure mask has shape [b, t]
            assert mask.dim() == 2 and mask.shape[0] == b and mask.shape[1] == t, (
                "mask must be [batch, seq_len]"
            )
            mask_f = mask.to(x.dtype)
        else:
            mask_f = None

        # Prepare ALiBi slopes (build in float32 for stability, then cast)
        slopes = _build_alibi_slopes(h, x.device, torch.float32).to(x.dtype)  # [h]

        # Learned decays
        lam = torch.sigmoid(self.decay_logit)  # [h, k]

        # Initialize associative memories and normalizers
        H_mem = torch.zeros(b, h, k_times, d, d, dtype=x.dtype, device=x.device)
        Z_norm = torch.zeros(b, h, k_times, 1, 1, dtype=x.dtype, device=x.device)

        # Local attention caches for previous chunk (up to window_size-1 tokens)
        w = self.window_size
        k_cache_prev = torch.zeros(b, h, 0, d, dtype=x.dtype, device=x.device)
        v_cache_prev = torch.zeros(b, h, 0, d, dtype=x.dtype, device=x.device)
        if mask_f is not None:
            m_cache_prev = torch.ones(b, 0, dtype=x.dtype, device=x.device)

        outputs = []  # list of [b, len_chunk, h, d]

        # Chunkwise processing
        for start in range(0, t, self.chunk_size):
            end = min(t, start + self.chunk_size)
            q_chunk = q[:, start:end]  # [b, tc, h, d]
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            gate_chunk = gate[:, start:end]  # [b, tc, h, k]
            alpha_chunk = alpha[:, start:end]  # [b, tc, h]
            if mask_f is not None:
                m_chunk = mask_f[:, start:end]  # [b, tc]

            tc = q_chunk.shape[1]
            out_chunk = []  # [tc x (b,h,d)]

            # Build full window source tensors for this chunk
            for i in range(tc):
                # Select current token projections
                q_t = q_chunk[:, i]  # [b, h, d]
                k_t = k_chunk[:, i]  # [b, h, d]
                v_t = v_chunk[:, i]  # [b, h, d]
                g_t = gate_chunk[:, i]  # [b, h, k]
                a_t = alpha_chunk[:, i]  # [b, h]

                # Apply mask to gate to avoid updating on padding positions
                if mask_f is not None:
                    m_t = m_chunk[:, i]  # [b]
                    g_t = g_t * rearrange(m_t, "b -> b 1 1")  # [b,h,k]

                # Update associative memory per timescale
                # Outer product: [b,h,d,d]
                outer = torch.einsum("bhi,bhj->bhij", k_t, v_t)
                # Expand for timescales: [b,h,1,d,d]
                outer_k = outer.unsqueeze(2)
                # Decay factors broadcast: [1,h,k,1,1]
                lam_b = rearrange(lam, "h k -> 1 h k 1 1")
                g_t_b = rearrange(g_t, "b h k -> b h k 1 1")

                H_mem = H_mem * lam_b + g_t_b * outer_k

                # Normalizer update
                s_t = k_t.abs().mean(dim=-1, keepdim=True)  # [b,h,1]
                s_t = s_t + self.eps
                s_t_b = rearrange(s_t, "b h 1 -> b h 1 1 1")  # [b,h,1,1,1]
                Z_norm = Z_norm * lam_b + g_t_b * s_t_b  # [b,h,k,1,1]

                # Linear-memory readout, normalized
                # y_k: [b,h,k,d]
                y_k = torch.einsum("bhi,bhkij->bhkj", q_t, H_mem)
                z_denom = Z_norm.squeeze(-1).squeeze(-1) + self.eps  # [b,h,k]
                y_k = y_k / z_denom.unsqueeze(-1)
                y_linear = y_k.sum(dim=2)  # [b,h,d]

                # Local causal attention over a fixed window with ALiBi
                # Compute how many tokens to take from previous cache and current chunk prefix (avoid quadratic cat)
                prev_len = k_cache_prev.shape[2]
                past_total = min(max(w - 1, 0), prev_len + i)
                prev_take = min(prev_len, past_total)
                curr_take = past_total - prev_take

                parts_k = []
                parts_v = []
                if prev_take > 0:
                    parts_k.append(k_cache_prev[:, :, -prev_take:])  # [b,h,prev_take,d]
                    parts_v.append(v_cache_prev[:, :, -prev_take:])
                if curr_take > 0:
                    k_curr_tail = rearrange(k_chunk[:, i - curr_take : i], "b l h d -> b h l d")
                    v_curr_tail = rearrange(v_chunk[:, i - curr_take : i], "b l h d -> b h l d")
                    parts_k.append(k_curr_tail)  # [b,h,curr_take,d]
                    parts_v.append(v_curr_tail)
                # Append current token
                k_curr = k_t.unsqueeze(2)  # [b,h,1,d]
                v_curr = v_t.unsqueeze(2)
                parts_k.append(k_curr)
                parts_v.append(v_curr)
                k_ctx = torch.cat(parts_k, dim=2)  # [b,h,L,d]
                v_ctx = torch.cat(parts_v, dim=2)

                if mask_f is not None:
                    parts_m = []
                    if prev_take > 0:
                        parts_m.append(m_cache_prev[:, -prev_take:])  # [b,prev_take]
                    if curr_take > 0:
                        parts_m.append(m_chunk[:, i - curr_take : i])  # [b,curr_take]
                    parts_m.append(m_chunk[:, i : i + 1])  # [b,1] current token
                    m_ctx = torch.cat(parts_m, dim=1)  # [b,L]

                # Compute attention scores: [b,h,L]
                scores = torch.einsum("bhd,bhld->bhl", q_t, k_ctx)

                # ALiBi bias: more negative for older positions
                # distances: [L], 0=current (last), increasing toward older tokens
                L = k_ctx.shape[2]
                # build distances in integer (always supported) then cast
                distances = torch.arange(L, device=x.device, dtype=torch.int64)
                distances = rearrange(distances, "l -> 1 1 l")  # [1,1,L]
                # Reverse so that current token is at the last index; distance = (L-1 - idx)
                distances = (L - 1) - distances  # [1,1,L]
                distances = distances.to(slopes.dtype)
                alibi = -rearrange(slopes, "h -> 1 h 1") * distances  # [1,h,L]
                scores = scores + alibi

                # Apply mask to local attention (safe softmax without NaNs when all masked)
                if mask_f is not None:
                    # m_ctx: [b,L] where 1 valid, 0 pad
                    attn_mask = rearrange(m_ctx > 0, "b l -> b 1 l")  # [b,1,L] bool
                    # Set invalid positions to -inf
                    masked_scores = scores.masked_fill(~attn_mask, float("-inf"))
                    # Safe softmax: handle rows with all -inf by zeroing attention
                    max_scores = masked_scores.max(dim=-1, keepdim=True).values  # [b,h,1]
                    # Replace non-finite max with zeros to avoid NaNs in exp
                    max_scores = torch.where(torch.isfinite(max_scores), max_scores, torch.zeros_like(max_scores))
                    exp_scores = torch.exp(masked_scores - max_scores) * attn_mask.to(masked_scores.dtype)
                    denom = exp_scores.sum(dim=-1, keepdim=True)  # [b,h,1]
                    attn = torch.where(denom > 0, exp_scores / (denom + 1e-20), torch.zeros_like(exp_scores))
                else:
                    attn = F.softmax(scores, dim=-1)

                y_local = torch.einsum("bhl,bhld->bhd", attn, v_ctx)

                # Mix local and linear branches
                y_t = (
                    rearrange(a_t, "b h -> b h 1") * y_local
                    + (1.0 - rearrange(a_t, "b h -> b h 1")) * y_linear
                )  # [b,h,d]

                out_chunk.append(y_t)

            # Stack outputs of the chunk
            out_chunk_t = torch.stack(out_chunk, dim=1)  # [b, tc, h, d]
            outputs.append(out_chunk_t)

            # Update caches for next chunk: keep last w-1 tokens of (prev + chunk)
            k_all = torch.cat([k_cache_prev, rearrange(k_chunk, "b t h d -> b h t d")], dim=2)  # [b,h,prev+tc,d]
            v_all = torch.cat([v_cache_prev, rearrange(v_chunk, "b t h d -> b h t d")], dim=2)
            keep = max(0, min(w - 1, k_all.shape[2]))
            if keep > 0:
                k_cache_prev = k_all[:, :, -keep:]
                v_cache_prev = v_all[:, :, -keep:]
            else:
                k_cache_prev = torch.zeros(b, h, 0, d, dtype=x.dtype, device=x.device)
                v_cache_prev = torch.zeros(b, h, 0, d, dtype=x.dtype, device=x.device)
            if mask_f is not None:
                m_all = torch.cat([m_cache_prev, m_chunk], dim=1)
                if keep > 0:
                    m_cache_prev = m_all[:, -keep:]
                else:
                    m_cache_prev = torch.ones(b, 0, dtype=x.dtype, device=x.device)

        # Concatenate all chunks: [b, t, h, d]
        y = torch.cat(outputs, dim=1)
        y = rearrange(y, "b t h d -> b t (h d)")

        # Output projection and residual
        y = self.out_proj(y)
        y = self.dropout(y)
        y = self.layer_norm(residual + y)
        return y


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block with evolved attention layer"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        ffn_hidden_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        self.attention = DeltaNet(hidden_size, num_heads, dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

        self.ffn_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x = self.attention(x, mask)

        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet model with evolved DeltaNet blocks"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [DeltaNetBlock(hidden_size, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        # Output layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t = input_ids.shape

        # Create position IDs
        position_ids = torch.arange(t, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(b, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final layer norm and projection
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Hybrid Linear + Local Attention Transformer
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Multi-timescale gated retention, normalized associative memory, local causal attention with ALiBi, chunkwise streaming
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Create a DeltaNet model with default parameters"""
    default_config = {
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "max_seq_len": 2048,
        "dropout": 0.1,
    }
    default_config.update(kwargs)
    return DeltaNetModel(vocab_size=vocab_size, **default_config)


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8)
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
