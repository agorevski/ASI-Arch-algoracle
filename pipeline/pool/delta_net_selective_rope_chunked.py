"""
DeltaNet: Evolved Linear Attention Architecture with Token-Selective Gating and RoPE
This implementation upgrades the baseline DeltaNet with:
- Token-conditional forgetting (beta), input gate (g), and output gate (o) per head
- Rotary positional embeddings (RoPE) applied to Q/K for improved relative position modeling
- Chunkwise causal scanning over the sequence for memory efficiency
- Vectorized multi-head processing (no Python loop over heads)
- RMSNorm pre-normalization for stability
- einsum-based delta rule with correct batched outer products and reads
- einops.rearrange replacing view/reshape everywhere
- Optional attention mask support (causal integrity preserved)
- @torch.compile applied to the core scan for performance

Complexity: O(B * H * L * Dh^2), sub-quadratic in sequence length.
"""

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, dim]
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def _rope_freqs(dim: int, seq_len: int, device, dtype, theta: float = 10000.0):
    # dim must be even for perfect pairing; if odd, last dim will remain unrotated
    half = dim // 2
    if half == 0:
        # Degenerate case
        cos = torch.ones(seq_len, 1, device=device, dtype=dtype)
        sin = torch.zeros(seq_len, 1, device=device, dtype=dtype)
        return cos, sin
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    # [seq_len, half]
    freqs = torch.einsum('l,d->ld', t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to last dimension.
    x: [B, H, L, D]
    cos/sin: [L, D//2]
    Returns same shape as x.
    """
    B, H, L, D = x.shape
    half = D // 2
    if half == 0:
        return x
    x1 = x[..., :half]
    x2 = x[..., half:half*2]
    # Broadcast cos/sin to [1,1,L,half]
    cos_ = cos.view(1, 1, L, half)
    sin_ = sin.view(1, 1, L, half)
    # Rotate pairs
    x1_ro = x1 * cos_ - x2 * sin_
    x2_ro = x1 * sin_ + x2 * cos_
    if D % 2 == 0:
        return torch.cat([x1_ro, x2_ro], dim=-1)
    else:
        # Append the last odd channel unchanged
        last = x[..., -1:].contiguous()
        return torch.cat([x1_ro, x2_ro, last], dim=-1)


@torch.compile(fullgraph=False, dynamic=True)
def delta_scan_chunked(q: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor,
                       beta: torch.Tensor,
                       in_gate: torch.Tensor,
                       out_gate: torch.Tensor,
                       attn_mask: Optional[torch.Tensor],
                       chunk_size: int) -> torch.Tensor:
    """
    Vectorized token-selective delta rule with chunkwise scanning.
    Shapes:
      q,k,v: [B, H, L, Dh]
      beta, in_gate, out_gate: [B, H, L] in [0,1]
      attn_mask: optional [B, L] with 1 for valid, 0 for pad
    Returns:
      y: [B, H, L, Dh]
    Causality is inherent by forward-only scan.
    """
    B, H, L, Dh = q.shape
    # Initialize associative memory per head: [B, H, Dh, Dh]
    M = torch.zeros(B, H, Dh, Dh, device=q.device, dtype=q.dtype)
    y = torch.zeros(B, H, L, Dh, device=q.device, dtype=q.dtype)

    # Prepare mask broadcast if provided (convert any non-zero to True)
    if attn_mask is not None:
        am = attn_mask.to(torch.bool)
    else:
        am = None

    # scaling for stability
    scale = 1.0 / math.sqrt(max(Dh, 1))

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        # Slice chunk
        q_c = q[:, :, start:end, :]  # [B,H,C,Dh]
        k_c = k[:, :, start:end, :]
        v_c = v[:, :, start:end, :]
        beta_c = beta[:, :, start:end]  # [B,H,C]
        g_c = in_gate[:, :, start:end]
        o_c = out_gate[:, :, start:end]
        if am is not None:
            am_c = am[:, start:end]  # [B,C]
        else:
            am_c = None

        C = end - start
        for t in range(C):
            # gates for step t
            f_t = beta_c[:, :, t]  # [B,H]
            g_t = g_c[:, :, t]
            o_t = o_c[:, :, t]

            # Forget previous memory (guarded by mask if provided)
            f_t_full = f_t.unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]

            if am_c is not None:
                # [B,1,1,1] broadcast over H and Dh dims
                m_t = am_c[:, t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                # Apply forget only on valid tokens; keep M unchanged on pads
                M = M * torch.where(m_t, f_t_full, torch.ones_like(f_t_full))
            else:
                M = M * f_t_full

            # Outer product update with input gate and scaling
            k_t = k_c[:, :, t, :]  # [B,H,Dh]
            v_t = v_c[:, :, t, :]  # [B,H,Dh]
            upd = torch.einsum('bhd,bhe->bhde', k_t, v_t)  # [B,H,Dh,Dh]
            upd_scaled = (g_t.unsqueeze(-1).unsqueeze(-1) * scale) * upd
            if am_c is not None:
                # Only write on valid tokens
                upd_scaled = upd_scaled * m_t
            M = M + upd_scaled

            # Readout: y_t = (q_t @ M) * out_gate; zero out padded outputs
            q_t = q_c[:, :, t, :]  # [B,H,Dh]
            y_t = torch.einsum('bhd,bhde->bhe', q_t, M)  # [B,H,Dh]
            if am_c is not None:
                o_t = o_t * am_c[:, t].unsqueeze(1)
            y[:, :, start + t, :] = y_t * o_t.unsqueeze(-1)

    return y


class DeltaNet(nn.Module):
    """
    Single DeltaNet attention layer with token-conditional gating,
    chunkwise scanning, and RoPE-enhanced Q/K.
    """

    def __init__(self, hidden_size: Optional[int] = None, d_model: Optional[int] = None,
                 num_heads: int = 8, dropout: float = 0.1,
                 chunk_size: int = 128, rope_theta: float = 10000.0, **kwargs):
        super().__init__()
        # Support both hidden_size and d_model for interface compatibility
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNet requires hidden_size or d_model to be specified")
        if hidden_size is None:
            hidden_size = d_model  # alias
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")

        self.chunk_size = int(chunk_size)
        self.rope_theta = float(rope_theta)

        # PreNorm for stability
        self.pre_norm = RMSNorm(hidden_size)

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Token-conditional gates per head
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)   # forgetting gate
        self.in_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)  # input modulation
        self.out_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)  # output modulation

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Initialize gates to favor retention but allow flow
        nn.init.constant_(self.beta_proj.bias, 2.0)   # sigmoid(2) ~ 0.88 (slow forgetting)
        nn.init.constant_(self.in_gate_proj.bias, 1.0)  # sigmoid(1) ~ 0.73
        nn.init.constant_(self.out_gate_proj.bias, 1.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            mask: Optional [B, L] with 1 for tokens to attend (non-pad)
        Returns:
            [B, L, D]
        """
        B, L, D = x.shape
        residual = x

        # Normalize mask shapes to [B, L] if provided
        if mask is not None:
            if mask.dim() == 4:
                # Common additive/attn mask shape [B, 1, 1, L]
                mask = mask.squeeze(1).squeeze(1)
            elif mask.dim() == 3:
                # Shape [B, 1, L]
                mask = mask.squeeze(1)
            # Ensure shape is [B, L]
            assert mask.shape == (B, L), f"Expected mask shape [B, L], got {tuple(mask.shape)}"

        # PreNorm
        x = self.pre_norm(x)

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Head split: [B, L, H, Dh] -> [B, H, L, Dh]
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # RoPE on Q/K
        cos, sin = _rope_freqs(self.head_dim, L, q.device, q.dtype, theta=self.rope_theta)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Gates in [0,1]
        beta = torch.sigmoid(self.beta_proj(x))  # [B,L,H]
        in_gate = torch.sigmoid(self.in_gate_proj(x))  # [B,L,H]
        out_gate = torch.sigmoid(self.out_gate_proj(x))  # [B,L,H]
        beta = rearrange(beta, 'b l h -> b h l')
        in_gate = rearrange(in_gate, 'b l h -> b h l')
        out_gate = rearrange(out_gate, 'b l h -> b h l')

        # Chunkwise delta scan
        y = delta_scan_chunked(q, k, v, beta, in_gate, out_gate, mask, self.chunk_size)  # [B,H,L,Dh]

        # Merge heads and output proj
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.out_proj(y)
        y = self.dropout(y)

        # Residual connection
        return residual + y


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block"""

    def __init__(self, hidden_size: Optional[int] = None, d_model: Optional[int] = None,
                 num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        # Support both hidden_size and d_model
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNetBlock requires hidden_size or d_model to be specified")
        if hidden_size is None:
            hidden_size = d_model

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        self.attention = DeltaNet(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        # Feed-forward network (GEGLU could be used; keep GELU for simplicity)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (residual inside)
        x = self.attention(x, mask)
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(self.ffn_layer_norm(x))
        x = residual + x
        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet model"""

    def __init__(self, vocab_size: int, hidden_size: Optional[int] = None, d_model: Optional[int] = None,
                 num_layers: int = 6, num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        # Support both hidden_size and d_model
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNetModel requires hidden_size or d_model to be specified")
        if hidden_size is None:
            hidden_size = d_model

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.layer_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = input_ids.shape
        # Create position IDs
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(B, -1)

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
DeltaNet Architecture Summary:
- Model Type: Linear Attention Transformer (Token-Selective Delta Rule)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Token-conditional forgetting/input/output gates, RoPE, chunkwise scan, RMSNorm
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    default_config = {
        'hidden_size': 512,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 2048,
        'dropout': 0.1
    }
    default_config.update(kwargs)
    return DeltaNetModel(vocab_size=vocab_size, **default_config)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attn_mask = torch.ones(batch_size, seq_len)
    with torch.no_grad():
        logits = model(input_ids, attn_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
