"""
DeltaNet: Evolved Linear Attention Architecture with Adaptive Gating, RoPE, and Causal Normalization
This implementation upgrades the baseline with:
- Content-adaptive per-head forget gates
- Rotary Position Embeddings (RoPE) applied to Q/K
- Causal normalization with positive feature maps (ELU+1) for stability
- Chunkwise processing and compiled inner scan for throughput
- Strict batch-size agnostic operations and einops.rearrange instead of view/reshape
Maintains sub-quadratic complexity and causal integrity.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _phi_positive(x: torch.Tensor) -> torch.Tensor:
    """Positive feature map to stabilize causal normalization."""
    return F.elu(x) + 1.0


def _build_rope_cache(seq_len: int, dim: int, device: torch.device, base: float = 10000.0):
    """Create RoPE cos/sin cache for given sequence length and head dimension.
    Supports odd dim by rotating the first 2*(dim//2) features and leaving the last one unchanged.
    """
    half = dim // 2
    if half == 0:
        # No rotation possible, return zeros to avoid NaNs; caller should handle identity
        cos = torch.ones((seq_len, 0), device=device, dtype=torch.float32)
        sin = torch.zeros((seq_len, 0), device=device, dtype=torch.float32)
        return cos, sin
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('t,f->tf', t, inv_freq)  # [T, half]
    cos = torch.cos(freqs).to(device)
    sin = torch.sin(freqs).to(device)
    return cos, sin  # [T, half] each


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to last dimension of x.
    x: [B, T, H, D]
    cos/sin: [T, D//2]
    Supports odd D by leaving the last feature unrotated.
    """
    B, T, H, D = x.shape
    half = D // 2
    if half == 0:
        return x
    main_dim = half * 2
    x_main = x[..., :main_dim]
    x_rem = x[..., main_dim:]
    x1, x2 = x_main[..., :half], x_main[..., half:]
    # expand cos/sin to [1, T, 1, half]
    cos_e = rearrange(cos, 't d -> 1 t 1 d').to(dtype=x.dtype, device=x.device)
    sin_e = rearrange(sin, 't d -> 1 t 1 d').to(dtype=x.dtype, device=x.device)
    x_rot1 = x1 * cos_e - x2 * sin_e
    x_rot2 = x2 * cos_e + x1 * sin_e
    x_rot = torch.cat([x_rot1, x_rot2], dim=-1)
    if x_rem.numel() == 0:
        return x_rot
    return torch.cat([x_rot, x_rem], dim=-1)


# -----------------------------------------------------------------------------
# Compiled core scan (per chunk)
# -----------------------------------------------------------------------------

# Define compiled function conditionally to avoid environments without torch.compile
if hasattr(torch, 'compile'):
    def _compile_decorator(fn):
        return torch.compile(fn, fullgraph=False, dynamic=True)
else:
    def _compile_decorator(fn):
        return fn


@_compile_decorator
def _delta_scan_chunk(
    q_chunk: torch.Tensor,  # [B, T, H, D]
    k_chunk: torch.Tensor,  # [B, T, H, D]
    v_chunk: torch.Tensor,  # [B, T, H, D]
    f_chunk: torch.Tensor,  # [B, T, H] in [0,1]
    S: torch.Tensor,        # [B, H, D, D]
    m: torch.Tensor,        # [B, H, D]
    eps: float,
    mask_chunk: Optional[torch.Tensor] = None,  # [B, T] with 1 for valid, 0 for pad
):
    """Causal delta-rule scan over a chunk with causal normalization.
    Returns outputs [B, T, H, D], final S [B, H, D, D], final m [B, H, D].
    """
    B, T, H, D = q_chunk.shape
    # Preallocate output tensor to avoid Python list appends inside compiled graph
    out_chunk = q_chunk.new_empty((B, T, H, D))
    for t in range(T):
        q_t = q_chunk[:, t]  # [B, H, D]
        k_t = k_chunk[:, t]  # [B, H, D]
        v_t = v_chunk[:, t]  # [B, H, D]
        f_t = f_chunk[:, t]  # [B, H]

        if mask_chunk is not None:
            mask_t = mask_chunk[:, t].to(q_t.dtype)  # [B]
            # Broadcast to [B, H, 1] for q/k/v, and [B, H] for f
            mask_bh1 = rearrange(mask_t, 'b -> b 1 1')
            mask_bh = rearrange(mask_t, 'b -> b 1')
            # Zero-out invalid positions contributions, and keep previous state unchanged by setting f=1 when mask=0
            q_t = q_t * mask_bh1
            k_t = k_t * mask_bh1
            v_t = v_t * mask_bh1
            f_t = f_t * mask_bh + (1.0 - mask_bh)

        # Decay previous state
        S = S * rearrange(f_t, 'b h -> b h 1 1')  # [B,H,D,D]
        m = m * rearrange(f_t, 'b h -> b h 1')    # [B,H,D]

        # Update associative state and normalizer
        outer = torch.einsum('bhd,bhe->bhde', k_t, v_t)  # [B,H,D,D]
        S = S + outer
        m = m + k_t  # [B,H,D]

        # Readout with causal normalization
        numer = torch.einsum('bhd,bhde->bhe', q_t, S)  # [B,H,D]
        denom = torch.einsum('bhd,bhd->bh', q_t, m).clamp_min(eps)  # [B,H]
        o_t = numer / denom.unsqueeze(-1)  # [B,H,D]
        out_chunk[:, t] = o_t

    return out_chunk, S, m  # [B,T,H,D], [B,H,D,D], [B,H,D]


# -----------------------------------------------------------------------------
# Core Layer: DeltaNet (attention/memory layer)
# -----------------------------------------------------------------------------

class DeltaNet(nn.Module):
    """DeltaNet attention layer with adaptive gating, RoPE, and causal normalization.

    Forward signature preserved: forward(x: [B, T, C], mask: Optional[[B, T]]).
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
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)  # per-head forget gate
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Defaults and hyperparameters
        self.chunk_size = kwargs.get('chunk_size', 256)
        self.rope_base = kwargs.get('rope_base', 10000.0)
        self.eps = kwargs.get('eps', 1e-6)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Initialize forget gate bias to target a relatively long half-life
        # f ≈ exp(-1/τ0) with τ0 ~ 512 tokens
        tau0 = 512.0
        f0 = math.exp(-1.0 / tau0)
        b0 = math.log(f0 / (1.0 - f0))
        nn.init.constant_(self.gate_proj.bias, b0)
        nn.init.normal_(self.gate_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through DeltaNet attention layer.

        Args:
            x: [B, T, C]
            mask: Optional [B, T] with 1 for valid tokens, 0 for padding
        Returns:
            [B, T, C]
        """
        B, T, C = x.shape
        residual = x

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads: [B, T, H, D]
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # Apply RoPE to Q/K
        cos, sin = _build_rope_cache(T, self.head_dim, x.device, base=self.rope_base)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Positive feature maps for causal normalization
        q_phi = _phi_positive(q)
        k_phi = _phi_positive(k)

        # Adaptive forget gate per head per time
        f = torch.sigmoid(self.gate_proj(x))  # [B, T, H]
        f = f.clamp(0.01, 0.999)  # prevent degeneracy

        # Chunkwise processing
        H = self.num_heads
        D = self.head_dim
        outputs = []

        # Initialize state per batch and head
        S = torch.zeros((B, H, D, D), device=x.device, dtype=x.dtype)
        m = torch.zeros((B, H, D), device=x.device, dtype=x.dtype)

        # Iterate over chunks
        for start in range(0, T, self.chunk_size):
            end = min(T, start + self.chunk_size)
            q_c = q_phi[:, start:end]  # [B, Tc, H, D]
            k_c = k_phi[:, start:end]
            v_c = v[:, start:end]
            f_c = f[:, start:end]
            mask_c = mask[:, start:end] if mask is not None else None

            o_c, S, m = _delta_scan_chunk(q_c, k_c, v_c, f_c, S, m, self.eps, mask_c)
            outputs.append(o_c)

        # Concatenate over time and merge heads
        out = torch.cat(outputs, dim=1)  # [B, T, H, D]
        out = rearrange(out, 'b t h d -> b t (h d)')

        # Output projection + dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection and layer norm
        out = self.layer_norm(residual + out)
        return out


# -----------------------------------------------------------------------------
# Transformer Block and Model (updated to use DeltaNet layer)
# -----------------------------------------------------------------------------

class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block."""

    def __init__(self, hidden_size: int, num_heads: int = 8,
                 ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        self.attention = DeltaNet(hidden_size, num_heads, dropout)
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
    """Complete DeltaNet model."""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(B, -1)

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
DeltaNet Architecture Summary:
- Model Type: Linear Attention Transformer (Delta + Causal Normalization)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Adaptive forget gates, RoPE, causal normalization, chunked scan
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model

def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    default_config = {
        'hidden_size': 512,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 2048,
        'dropout': 0.1,
    }
    default_config.update(kwargs)
    return DeltaNetModel(vocab_size=vocab_size, **default_config)


if __name__ == "__main__":
    # Quick functional test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4)
    B, T = 2, 100
    input_ids = torch.randint(0, 1000, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
