"""
DeltaNet: Gated Multi-Scale Linear Attention with RoPE and Chunkwise Processing
Evolution of the baseline architecture to address long-range retention, positional bias,
and sharp local interactions while preserving sub-quadratic complexity.
"""

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------
# Normalization and Utilities
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return x_normed * self.weight


class LayerScale(nn.Module):
    def __init__(self, d: int, init_value: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# -----------------------------
# Rotary Positional Embeddings
# -----------------------------

def _rope_frequencies(dim: int, base: float = 10000.0, device=None, dtype=None) -> torch.Tensor:
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    return inv_freq  # [half]


def _rope_angles(seq_len: int, dim: int, device, dtype, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE angles safely.
    Always compute cos/sin in float32 for numerical stability and CPU support,
    then cast to the target dtype at the end.
    """
    assert dim % 2 == 0, "RoPE requires even head dimension"
    calc_dtype = torch.float32
    inv_freq = _rope_frequencies(dim, base=base, device=device, dtype=calc_dtype)  # [half]
    pos = torch.arange(seq_len, device=device, dtype=calc_dtype)  # [T]
    freqs = torch.einsum('t,f->tf', pos, inv_freq)  # [T, half]
    # duplicate for sin/cos interleaving across last dim
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # expand to full dim by interleaving
    cos_full = rearrange(torch.stack([cos, cos], dim=-1), 't f p -> t (f p)')  # [T, dim]
    sin_full = rearrange(torch.stack([sin, sin], dim=-1), 't f p -> t (f p)')  # [T, dim]
    # cast to target dtype
    cos_full = cos_full.to(dtype)
    sin_full = sin_full.to(dtype)
    return cos_full, sin_full


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, T, H, D]
    B, T, H, D = q.shape
    device = q.device
    dtype = q.dtype
    cos, sin = _rope_angles(T, D, device=device, dtype=dtype)
    # [T, D] -> [1, T, 1, D]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # rotate pairs: (x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]
        x_rot2 = x2 * cos[..., ::2] + x1 * sin[..., ::2]
        # stack on new last dim (pair), then interleave back to full D
        return rearrange(
            torch.stack([x_rot1, x_rot2], dim=-1),
            '... d p -> ... (d p)'
        )

    return _rotate(q), _rotate(k)


# -----------------------------
# Core DeltaNet Attention Layer
# -----------------------------
class DeltaNetLayer(nn.Module):
    """Gated multi-scale linear attention with RoPE and local windowed attention.
    - Linear-time delta memory with per-head gates (retention-biased)
    - RoPE applied to q/k for relative positional inductive bias
    - Small sliding-window local attention fused via learned gate
    - Chunked sequential processing for memory efficiency and stability
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        )

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Gates: per-head retention gate z in [0,1], initialized to retain (bias ~ +2)
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # Fusion gate between local and linear outputs per head
        self.fuse_proj = nn.Linear(hidden_size, num_heads, bias=True)
        nn.init.constant_(self.fuse_proj.bias, 0.0)

        # Normalization and residual scaling (Pre-Norm)
        self.norm = RMSNorm(hidden_size)
        self.res_scale = LayerScale(hidden_size, init_value=1e-3)

        # Hyper-parameters
        self.dropout = nn.Dropout(dropout)
        self.chunk_size = 128
        self.window_size = 64
        self.scale = self.head_dim ** -0.5

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @torch.compile(mode="reduce-overhead", fullgraph=False)
    def _gated_delta_memory(self,
                            q: torch.Tensor,
                            k: torch.Tensor,
                            v: torch.Tensor,
                            gate: torch.Tensor,
                            attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Chunked gated delta memory computation.
        Args:
            q,k,v: [B, T, H, D]
            gate: [B, T, H] in [0,1] retention gate
            attn_mask: Optional [B, T] with 1 for valid, 0 for pad
        Returns:
            y: [B, T, H, D]
        """
        B, T, H, D = q.shape
        # State per head: [B, H, D, D]
        state = q.new_zeros((B, H, D, D))
        outputs = []
        chunk = self.chunk_size
        # Prepare mask on same device/dtype
        if attn_mask is None:
            mask = q.new_ones((B, T))
        else:
            mask = attn_mask.to(device=q.device, dtype=q.dtype)

        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            q_c = q[:, s:e]        # [B, Tc, H, D]
            k_c = k[:, s:e]
            v_c = v[:, s:e]
            g_c = gate[:, s:e]     # [B, Tc, H]
            m_c = mask[:, s:e]     # [B, Tc]
            Tc = q_c.shape[1]

            # step through chunk sequentially (keeps O(T))
            y_c = []
            for t in range(Tc):
                q_t = q_c[:, t]      # [B, H, D]
                k_t = k_c[:, t]
                v_t = v_c[:, t]
                g_t = g_c[:, t]      # [B, H]
                # masks for this timestep
                m_t = m_c[:, t].unsqueeze(-1).unsqueeze(-1)      # [B,1,1]
                m_t_outer = m_t.unsqueeze(-1)                    # [B,1,1,1]
                # effective gate: if padded, do not change state
                g_eff = g_t.unsqueeze(-1).unsqueeze(-1)          # [B,H,1,1]
                g_eff = g_eff * m_t_outer + (1.0 - m_t_outer)    # broadcast-safe

                state = state * g_eff  # retention
                # outer product update: [B,H,D,D]
                outer = torch.einsum('bhd,bhe->bhde', k_t, v_t)
                outer = outer * m_t_outer  # do not update on pad
                state = state + outer
                # output: q @ state -> [B,H,D]
                y_t = torch.einsum('bhd,bhde->bhe', q_t, state)
                # zero output for padded query
                y_t = y_t * m_t  # [B,1,1] broadcast over H,D
                y_c.append(y_t)
            y_c = torch.stack(y_c, dim=1)  # [B, Tc, H, D]
            outputs.append(y_c)
        y = torch.cat(outputs, dim=1)
        return y

    @torch.compile(mode="reduce-overhead", fullgraph=False)
    def _local_attention(self,
                         q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Chunked causal sliding-window attention.
        Args:
            q,k,v: [B, T, H, D]
            attn_mask: Optional [B, T] 1 for valid
        Returns:
            y: [B, T, H, D]
        """
        B, T, H, D = q.shape
        w = self.window_size
        chunk = self.chunk_size
        outputs = []
        if attn_mask is None:
            mask = q.new_ones((B, T))
        else:
            mask = attn_mask.to(device=q.device)

        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            q_c = q[:, s:e]  # [B, Tc, H, D]
            Tc = q_c.shape[1]
            # context indices
            ctx_start = max(0, s - w)
            k_ctx = k[:, ctx_start:e]  # [B, Tc+w', H, D]
            v_ctx = v[:, ctx_start:e]
            m_ctx = mask[:, ctx_start:e]  # [B, Tc+w']
            ctx_len = k_ctx.shape[1]

            # reshape to [B,H,*,D]
            q_bh = rearrange(q_c, 'b t h d -> b h t d')
            k_bh = rearrange(k_ctx, 'b s h d -> b h s d')
            v_bh = rearrange(v_ctx, 'b s h d -> b h s d')

            # attention scores: [B,H,Tc,ctx]
            scores = torch.einsum('bhtd,bhsd->bhts', q_bh, k_bh) * self.scale

            # causal mask within current chunk relative to context
            q_pos = torch.arange(s, e, device=q.device)  # [Tc]
            k_pos = torch.arange(ctx_start, e, device=q.device)  # [ctx]
            causal = (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)).to(scores.dtype)  # [Tc, ctx]
            # choose large negative compatible with dtype to avoid -inf in fp16
            big_neg = torch.tensor(1e9, device=scores.device, dtype=scores.dtype)
            if scores.dtype == torch.float16:
                big_neg = torch.tensor(1e4, device=scores.device, dtype=scores.dtype)
            scores = scores + (causal.unsqueeze(0).unsqueeze(0) - 1.0) * big_neg

            # mask padding on keys
            key_mask = m_ctx.to(dtype=scores.dtype)  # [B, ctx]
            scores = scores + (key_mask.unsqueeze(1).unsqueeze(2) - 1.0) * big_neg

            # softmax and attend; guard against NaNs for fully-masked rows
            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)

            y_bh = torch.einsum('bhts,bhsd->bhtd', attn, v_bh)  # [B,H,Tc,D]
            y_c = rearrange(y_bh, 'b h t d -> b t h d')

            # zero out padded queries
            q_mask = mask[:, s:e].to(dtype=y_c.dtype).unsqueeze(-1).unsqueeze(-1)
            y_c = y_c * q_mask

            outputs.append(y_c)

        y = torch.cat(outputs, dim=1)
        return y

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm
        residual = x
        x = self.norm(x)

        B, T, Dm = x.shape
        H = self.num_heads
        Dh = self.head_dim

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [B, T, H, Dh]
        q = rearrange(q, 'b t (h d) -> b t h d', h=H)
        k = rearrange(k, 'b t (h d) -> b t h d', h=H)
        v = rearrange(v, 'b t (h d) -> b t h d', h=H)

        # Apply RoPE to q/k
        q, k = apply_rope(q, k)

        # Gates
        gate_ret = torch.sigmoid(self.gate_proj(x))        # [B, T, H]
        gate_fuse = torch.sigmoid(self.fuse_proj(x))       # [B, T, H]

        # Linear delta memory branch
        y_mem = self._gated_delta_memory(q, k, v, gate_ret, mask)  # [B,T,H,Dh]

        # Local attention branch
        y_loc = self._local_attention(q, k, v, mask)  # [B,T,H,Dh]

        # Fuse branches per head: y = g*y_loc + (1-g)*y_mem
        g = gate_fuse.unsqueeze(-1)  # [B,T,H,1]
        y = g * y_loc + (1.0 - g) * y_mem

        # Merge heads and output projection
        y = rearrange(y, 'b t h d -> b t (h d)')
        y = self.out_proj(y)
        y = self.dropout(y)

        # Residual with LayerScale
        return residual + self.res_scale(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop2 = nn.Dropout(dropout)
        self.res_scale = LayerScale(d_model, init_value=1e-3)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return residual + self.res_scale(x)


class DeltaNetBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.attention = DeltaNetLayer(hidden_size, num_heads, dropout, **kwargs)
        self.ffn = FeedForward(hidden_size, ffn_hidden_size, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, mask)
        x = self.ffn(x)
        return x


class DeltaNet(nn.Module):
    """Complete DeltaNet model with RoPE, gated delta memory, and local attention."""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        # Keep absolute positional embedding module for compatibility; scale defaults to 0
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.abs_pos_scale = nn.Parameter(torch.tensor(0.0))  # disable absolute by default
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads, dropout=dropout, **kwargs)
            for _ in range(num_layers)
        ])

        # Output head
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape

        # Token embeddings
        tok = self.token_embedding(input_ids)

        # Absolute position (scaled, default 0) with safe indexing for T>max_seq_len
        pos_ids = torch.arange(T, device=input_ids.device)
        if T > self.max_seq_len:
            pos_ids = pos_ids % self.max_seq_len
        pos_ids = pos_ids.unsqueeze(0).expand(B, T)
        pos = self.position_embedding(pos_ids) * self.abs_pos_scale

        x = tok + pos
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Gated Linear Attention + Local Window
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {getattr(self.layers[0].attention, 'num_heads', 'N/A')}
- Max Sequence Length: {self.max_seq_len}
- Positional Encoding: RoPE on q/k (absolute embedding scaled by abs_pos_scale={self.abs_pos_scale.item():.3f})
- Memory: Per-head gated delta associative state (D x D per head)
- Local Attention: Sliding window size={getattr(self.layers[0].attention, 'window_size', 'N/A')}
- Chunk Size: {getattr(self.layers[0].attention, 'chunk_size', 'N/A')}
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Backward compatibility alias
DeltaNetModel = DeltaNet


# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNet:
    default_config = {
        'hidden_size': 512,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 2048,
        'dropout': 0.1
    }
    default_config.update(kwargs)
    return DeltaNet(vocab_size=vocab_size, **default_config)


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    b, t = 2, 100
    inp = torch.randint(0, 1000, (b, t))
    attn_mask = torch.ones(b, t)
    with torch.no_grad():
        out = model(inp, attn_mask)
        print(f"Input shape: {inp.shape}")
        print(f"Output shape: {out.shape}")
        print(model.get_architecture_summary())
