"""
DeltaNet: Normalized Linear Attention with Multi-Timescale Memory and RoPE
Evolved architecture implementing:
- Softmax-like normalization via denominator state (linearized softmax)
- Multi-timescale forgetting per head (2 channels) with learnable betas
- Rotary positional embeddings (RoPE) on Q/K
- Chunked causal processing with fp32 accumulators
- Batch-size agnostic operations with einops.rearrange
- Sub-quadratic complexity (O(n))

Maintains original model interfaces and integrates seamlessly with existing blocks.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------
# Rotary Positional Embeddings
# -----------------------------

def _build_rope_cache(seq_len: int, dim: int, device, dtype, base: float = 10000.0):
    """Build RoPE cos/sin cache that also supports odd head dimensions.
    Returns cos, sin with shapes [S, floor(D/2)].
    """
    half = dim // 2
    if half == 0:
        # No rotation possible (D=1); return empty caches on correct device/dtype
        empty = torch.empty(seq_len, 0, device=device, dtype=dtype)
        return empty, empty
    # use range up to 2*half to keep frequency scaling aligned with original dim
    inv_freq = 1.0 / (base ** (torch.arange(0, 2 * half, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('s,d->sd', t, inv_freq)  # [S, half]
    cos = torch.cos(freqs).to(dtype=dtype)
    sin = torch.sin(freqs).to(dtype=dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to last dimension of x (head_dim), given cos/sin of shape [S, floor(D/2)].
    x: [B, S, H, D]
    returns: [B, S, H, D]
    Supports odd D by leaving the last unpaired dimension unchanged.
    """
    B, S, H, D = x.shape
    half = D // 2
    if half == 0:
        return x
    # split first 2*half dims into two halves to rotate; leave any tail as-is
    x1 = x[..., :half]
    x2 = x[..., half: 2 * half]
    tail = x[..., 2 * half:]
    # reshape cos/sin to broadcast using einops
    cos_ = rearrange(cos, 's d -> 1 s 1 d')  # [1, S, 1, half]
    sin_ = rearrange(sin, 's d -> 1 s 1 d')  # [1, S, 1, half]
    x_rot1 = x1 * cos_ - x2 * sin_
    x_rot2 = x1 * sin_ + x2 * cos_
    if tail.numel() == 0:
        return torch.cat([x_rot1, x_rot2], dim=-1)
    else:
        return torch.cat([x_rot1, x_rot2, tail], dim=-1)


# ---------------------------------------
# Core chunk processor (compiled for speed)
# ---------------------------------------

@torch.compile
def _delta_chunk_forward(
    q_phi: torch.Tensor,  # [BH, Lc, D]
    k_phi: torch.Tensor,  # [BH, Lc, D]
    v: torch.Tensor,      # [BH, Lc, D]
    beta: torch.Tensor,   # [BH, Lc, C]
    mask: Optional[torch.Tensor],  # [BH, Lc] or None
    M_state: torch.Tensor,  # [BH, C, D, D] (fp32)
    Z_state: torch.Tensor,  # [BH, C, D] (fp32)
    eps: float = 1e-6,
):
    BH, Lc, D = q_phi.shape
    C = beta.shape[-1]
    outputs = []
    for t in range(Lc):
        beta_t = beta[:, t]  # [BH, C]
        # gating for padding mask
        if mask is not None:
            m_t = mask[:, t].to(q_phi.dtype)  # [BH]
        else:
            m_t = None

        # fetch current token features
        k_t = k_phi[:, t]  # [BH, D]
        q_t = q_phi[:, t]  # [BH, D]
        v_t = v[:, t]      # [BH, D]

        if m_t is not None:
            m_t_uns = m_t.unsqueeze(-1)  # [BH, 1]
            k_t = k_t * m_t_uns
            v_t = v_t * m_t_uns
            q_t = q_t * m_t_uns

        # Update states (fp32 accumulators)
        # decay per channel
        decay = beta_t.to(dtype=M_state.dtype).unsqueeze(-1).unsqueeze(-1)  # [BH, C, 1, 1]
        decay_z = beta_t.to(dtype=Z_state.dtype).unsqueeze(-1)  # [BH, C, 1]

        if m_t is not None:
            # Do not decay on masked (padded) steps to ensure pad-invariance
            mM = m_t.to(M_state.dtype).view(BH, 1, 1, 1)  # [BH,1,1,1]
            mZ = m_t.to(Z_state.dtype).view(BH, 1, 1)     # [BH,1,1]
            # effective_decay = 1 when mask==0, else beta when mask==1
            eff_decay_M = decay * mM + (1.0 - mM)
            eff_decay_Z = decay_z * mZ + (1.0 - mZ)
            M_state = M_state * eff_decay_M
            Z_state = Z_state * eff_decay_Z
        else:
            M_state = M_state * decay  # [BH, C, D, D]
            Z_state = Z_state * decay_z  # [BH, C, D]

        # outer product term shared across channels
        # outer: [BH, D, D]
        outer = torch.einsum('bd,be->bde', k_t.to(M_state.dtype), v_t.to(M_state.dtype))
        # add to each channel
        M_state = M_state + outer.unsqueeze(1)  # [BH, C, D, D]
        Z_state = Z_state + k_t.to(Z_state.dtype).unsqueeze(1)  # [BH, C, D]

        # compute output: numerator/denominator
        # num_j = q_t @ M_j -> [BH, C, D]
        num = torch.einsum('bd,bcde->bce', q_t.to(M_state.dtype), M_state)  # [BH, C, D]
        den = torch.einsum('bd,bcd->bc', q_t.to(Z_state.dtype), Z_state)    # [BH, C]
        # sum across channels
        num = num.sum(dim=1)  # [BH, D]
        den = den.sum(dim=1)  # [BH]
        den = den + eps
        y_t = (num / den.unsqueeze(-1)).to(v.dtype)  # [BH, D]
        outputs.append(y_t)

    return torch.stack(outputs, dim=1), M_state, Z_state  # [BH, Lc, D]


class DeltaNet(nn.Module):
    """Single DeltaNet attention layer with normalized linear attention and RoPE.

    Preserves forward signature: forward(x: [B, S, D], mask: Optional[[B, S]]) -> [B, S, D]
    """

    def __init__(self, hidden_size: Optional[int] = None, d_model: Optional[int] = None,
                 num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        # Support both hidden_size and d_model naming
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNet requires hidden_size or d_model to be specified")
        if hidden_size is None:
            hidden_size = d_model
        if d_model is None:
            d_model = hidden_size
        self.hidden_size = int(hidden_size)
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size, (
            f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}"
        )

        # Multi-timescale channels per head
        self.num_channels = 2  # default K=2 channels per head
        self.chunk_size = kwargs.get('chunk_size', 256)
        self.eps = 1e-6

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # Beta projection per head/channel
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads * self.num_channels, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Initialize betas to high retention (~0.98) via bias-only; weights zero for stability
        nn.init.zeros_(self.beta_proj.weight)
        with torch.no_grad():
            target_beta = 0.98
            bias_val = math.log(target_beta / (1.0 - target_beta))  # logit
            self.beta_proj.bias.fill_(bias_val)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        # Non-negative kernel feature map (linearized softmax): elu + 1
        return F.elu(x, alpha=1.0) + 1.0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Accept either mask or attention_mask keyword; prefer explicit mask if both provided
        if mask is None and attention_mask is not None:
            mask = attention_mask

        B, S, D_model = x.shape
        H = self.num_heads
        Dh = self.head_dim

        residual = x

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head: [B, S, H, Dh]
        q = rearrange(q, 'b s (h d) -> b s h d', h=H)
        k = rearrange(k, 'b s (h d) -> b s h d', h=H)
        v = rearrange(v, 'b s (h d) -> b s h d', h=H)

        # Apply RoPE to q and k
        cos, sin = _build_rope_cache(S, Dh, device=x.device, dtype=x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Compute kernel features
        q_phi = self._phi(q)
        k_phi = self._phi(k)

        # Compute betas per head/channel: [B, S, H, C]
        beta_raw = self.beta_proj(x)  # [B, S, H*C]
        beta = rearrange(beta_raw, 'b s (h c) -> b s h c', h=H, c=self.num_channels)
        beta = torch.sigmoid(beta)
        # Avoid extreme values
        beta = beta.clamp(0.001, 0.999)

        # Prepare mask
        if mask is not None:
            assert mask.dim() == 2 and mask.shape == (B, S), "mask must be [B, S]"
            mask_b = mask.to(dtype=q.dtype)
        else:
            mask_b = None

        # Flatten batch and heads for efficient compute
        q_phi = rearrange(q_phi, 'b s h d -> (b h) s d')  # [BH, S, Dh]
        k_phi = rearrange(k_phi, 'b s h d -> (b h) s d')
        v = rearrange(v, 'b s h d -> (b h) s d')
        beta = rearrange(beta, 'b s h c -> (b h) s c')       # [BH, S, C]
        if mask_b is not None:
            mask_flat = rearrange(mask_b, 'b s -> b 1 s')
            mask_flat = mask_flat.expand(B, H, S)
            mask_flat = rearrange(mask_flat, 'b h s -> (b h) s')  # [BH, S]
        else:
            mask_flat = None

        BH = q_phi.shape[0]
        C = self.num_channels

        # Initialize fp32 accumulators
        M_state = torch.zeros(BH, C, Dh, Dh, device=x.device, dtype=torch.float32)
        Z_state = torch.zeros(BH, C, Dh, device=x.device, dtype=torch.float32)

        # Chunked processing along sequence
        outputs_chunks = []
        for start in range(0, S, self.chunk_size):
            end = min(start + self.chunk_size, S)
            q_chunk = q_phi[:, start:end]
            k_chunk = k_phi[:, start:end]
            v_chunk = v[:, start:end]
            beta_chunk = beta[:, start:end]
            mask_chunk = mask_flat[:, start:end] if mask_flat is not None else None

            out_chunk, M_state, Z_state = _delta_chunk_forward(
                q_chunk, k_chunk, v_chunk, beta_chunk, mask_chunk, M_state, Z_state, self.eps
            )
            outputs_chunks.append(out_chunk)

        # Concatenate chunks and restore shapes
        out = torch.cat(outputs_chunks, dim=1)  # [BH, S, Dh]
        out = rearrange(out, '(b h) s d -> b s h d', b=B, h=H)
        out = rearrange(out, 'b s h d -> b s (h d)')  # [B, S, D_model]

        # Output projection, dropout, residual, norm
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        return out


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block"""

    def __init__(self, hidden_size: int, num_heads: int = 8,
                 ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        # Fix: pass arguments to DeltaNet using keywords to avoid mis-ordering
        self.attention = DeltaNet(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout)
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
    """Complete DeltaNet model"""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
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

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S = input_ids.shape

        # Create position IDs (safe for any sequence length via wrap-around)
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        if S > self.max_seq_len:
            # Wrap positions to stay within embedding table range to avoid OOB indexing
            position_ids = position_ids % self.max_seq_len

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
- Model Type: Normalized Linear Attention Transformer with RoPE
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Normalized linear attention (num/den), multi-timescale (K=2), RoPE
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
    # Test the model
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8)
    model.eval()

    # Create sample input
    batch_size, seq_len = 3, 2300  # test beyond max_seq_len to verify safety
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[0, 90:] = 0  # add some padding

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
