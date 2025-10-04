"""
DeltaNet: Evolved Linear Attention Architecture with Multi-Timescale Delta Memory
- Implements chunkwise, causal, sub-quadratic computation
- Fuses multi-head processing and introduces multi-timescale memory per head
- Stabilizes with positive feature maps (elu+1) for Q/K and float32 state accumulation
- Adds input-conditioned retention gating and per-scale base retentions (log-spaced)
- Batch-size agnostic and uses einops.rearrange for all reshape operations
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def _logit(x: float) -> float:
    x = min(max(x, 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


class DeltaNet(nn.Module):
    """Single DeltaNet attention layer (evolved)
    - Preserves forward signature: forward(x: Tensor, mask: Optional[Tensor]) -> Tensor
    - Maintains sub-quadratic complexity (O(n) per layer)
    - Uses chunked, causal processing and fused multi-head updates
    - Multi-timescale memory per head with K states (default 4)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        )

        # Multi-timescale parameters
        self.num_scales = 4  # K states per head (sensible default)
        # Base retentions per scale, roughly targeting time constants ~[16, 64, 256, 1024]
        base_retentions = [math.exp(-1.0 / t) for t in [16.0, 64.0, 256.0, 1024.0]]
        # Learnable per-head per-scale base retention logits initialized near these values
        init_logits = torch.tensor([_logit(r) for r in base_retentions], dtype=torch.float32)
        init_logits = repeat(init_logits, 'k -> h k', h=self.num_heads)
        self.base_retention_logit = nn.Parameter(init_logits)  # [H, K]

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Input-conditioned retention gate per head
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        # Per-head per-scale mixing weights for output combination
        self.mix_proj = nn.Linear(hidden_size, self.num_heads * self.num_scales, bias=True)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

        # Chunk size for processing (sensible default; can be tuned)
        self.chunk_size = 128

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Initialize beta bias to favor moderate retention r ~ 0.99
        if self.beta_proj.bias is not None:
            nn.init.constant_(self.beta_proj.bias, _logit(0.99))

        # Initialize mix proj to near-uniform across scales
        nn.init.zeros_(self.mix_proj.bias)
        nn.init.xavier_uniform_(self.mix_proj.weight)

    @torch.compile  # compile core scan for performance
    def _delta_scan(
        self,
        q: torch.Tensor,  # [B, T, H, D]
        k: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, H, D]
        beta: torch.Tensor,  # [B, T, H] in (0,1)
        mix_logits: torch.Tensor,  # [B, T, H, K]
        mask: Optional[torch.Tensor] = None,  # [B, T] or [B, T, 1]
        chunk_size: int = 128,
    ) -> torch.Tensor:
        B, T, H, D = q.shape
        K = mix_logits.shape[-1]

        # Prepare mask
        if mask is None:
            m = torch.ones((B, T), device=q.device, dtype=q.dtype)
        else:
            if mask.dim() == 3:
                m = mask.squeeze(-1)
            else:
                m = mask
            # Ensure mask is on the same device and dtype as q for correct broadcasting/ops
            m = m.to(device=q.device, dtype=q.dtype)
        # Expand mask with enough singleton dims so that m[:, t] broadcasts to [B, H, K, D, D]
        m = rearrange(m, 'b t -> b t 1 1 1 1')  # [B, T, 1, 1, 1, 1]

        # States: [B, H, K, D, D], accumulate in float32 for stability
        state = torch.zeros((B, H, K, D, D), device=q.device, dtype=torch.float32)

        # Preallocate output for compile-friendliness (avoid Python list appends)
        out_all = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)

        # Precompute per-head per-scale base retentions (sigmoid to (0,1))
        base_r = torch.sigmoid(self.base_retention_logit)  # [H, K]
        base_r = rearrange(base_r, 'h k -> 1 h k 1 1')  # [1, H, K, 1, 1]

        # Process in causal chunks; within each chunk, strict left-to-right updates
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)

            q_chunk = q[:, start:end]  # [B, Tc, H, D]
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            beta_chunk = beta[:, start:end]  # [B, Tc, H]
            mix_chunk = mix_logits[:, start:end]  # [B, Tc, H, K]
            m_chunk = m[:, start:end]  # [B, Tc, 1, 1, 1, 1]

            Tc = q_chunk.shape[1]

            for t in range(Tc):
                q_t = q_chunk[:, t]  # [B, H, D]
                k_t = k_chunk[:, t]
                v_t = v_chunk[:, t]
                beta_t = beta_chunk[:, t]  # [B, H]
                mix_t = mix_chunk[:, t]  # [B, H, K]
                m_t = m_chunk[:, t]  # [B, 1, 1, 1, 1]

                # Dynamic retention per head per scale: beta_t already in (0,1)
                dyn_r = rearrange(beta_t, 'b h -> b h 1 1 1')  # [B, H, 1, 1, 1]
                # Combine base and dynamic: r_eff in (0,1)
                r_eff = torch.clamp(dyn_r * base_r, 0.0, 0.999995)  # [B, H, K, 1, 1]

                # Apply forgetting (masked positions do not advance state)
                state = torch.where(m_t > 0, state * r_eff.to(dtype=state.dtype), state)

                # Outer product update per head: k_t^T v_t -> [B, H, D, D]
                # Use float32 accumulation
                outer = torch.einsum('bhd,bhe->bhde', k_t.to(torch.float32), v_t.to(torch.float32))
                # Broadcast to scales
                outer = rearrange(outer, 'b h d e -> b h 1 d e')  # [B, H, 1, D, E]

                # Masked positions don't write
                state = torch.where(m_t > 0, state + outer, state)

                # Compute per-scale outputs: q_t @ state -> [B, H, K, D]
                # q_t: [B, H, D], state: [B, H, K, D, D]
                o_scales = torch.einsum('bhd,bhkde->bhke', q_t.to(torch.float32), state)

                # Mix across scales with softmax over K
                w = F.softmax(mix_t, dim=-1)  # [B, H, K]
                w = rearrange(w, 'b h k -> b h k 1')
                o_t = torch.sum(w * o_scales, dim=2)  # [B, H, D]

                # Zero outputs at masked positions per batch (broadcast over heads and dims)
                cond = (m_t > 0)  # [B, 1, 1, 1, 1] (bool)
                cond = cond.reshape(B, 1, 1)  # [B, 1, 1]
                o_t = o_t * cond.to(dtype=o_t.dtype)

                out_all[:, start + t] = o_t.to(dtype=q.dtype)

        return out_all

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass
        Args:
            x: [batch, seq_len, hidden_size]
            mask: Optional attention mask [batch, seq_len] (1 for valid, 0 for pad)
        Returns:
            [batch, seq_len, hidden_size]
        """
        # Residual
        residual = x

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # Stabilize with positive feature maps (elu+1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Dynamic retention gate per head
        beta = torch.sigmoid(self.beta_proj(x))  # [B, T, H]

        # Per-scale mixing logits
        mix_logits = self.mix_proj(x)  # [B, T, H*K]
        mix_logits = rearrange(mix_logits, 'b t (h k) -> b t h k', h=self.num_heads, k=self.num_scales)

        # Chunked causal multi-timescale delta scan
        head_out = self._delta_scan(q, k, v, beta, mix_logits, mask=mask, chunk_size=self.chunk_size)

        # Merge heads and project
        out = rearrange(head_out, 'b t h d -> b t (h d)')
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual + LayerNorm
        out = self.layer_norm(residual + out)
        return out


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block"""

    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
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
        x = self.attention(x, mask)
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet model"""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6, num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Tie weights

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape

        # Position IDs
        if T > self.max_seq_len:
            # Expand position embedding on-the-fly to accommodate longer sequences
            new_pos = nn.Embedding(T, self.hidden_size, device=input_ids.device)
            # Initialize new positions from old (copy) and random for the rest
            with torch.no_grad():
                new_pos.weight[: self.max_seq_len].copy_(self.position_embedding.weight)
                nn.init.normal_(new_pos.weight[self.max_seq_len :], std=0.02)
            self.position_embedding = new_pos
            self.max_seq_len = T

        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Linear Attention Transformer with Multi-Timescale Delta Memory
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Multi-scale memory, input-conditioned retention, chunked causal scan, positive feature maps
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model
def create_model(vocab_size: int = 50257, hidden_size: int = 512, num_layers: int = 6, num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1) -> DeltaNetModel:
    return DeltaNetModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Simple smoke test
    torch.set_float32_matmul_precision('high')
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8)
    B, T = 3, 97
    input_ids = torch.randint(0, 1000, (B, T))
    attention_mask = torch.ones(B, T)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
