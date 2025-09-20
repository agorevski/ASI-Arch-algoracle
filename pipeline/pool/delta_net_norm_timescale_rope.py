"""
DeltaNet: Evolved Linear Attention Architecture with Normalized Multi-Timescale Memory
This implementation upgrades the baseline with:
- Normalized linear attention accumulators (kernelized softmax with feature map phi)
- Multi-timescale content-dependent forgetting per head (default M=2)
- Chunked causal streaming for sub-quadratic complexity and memory efficiency
- Optional RoPE positional mixing applied to q/k for better extrapolation
- einops.rearrange used universally for robust shape handling

Interfaces are preserved:
- Layer class name: DeltaNet (forward: (x: [b t d], mask: Optional[b t]))
- Block, Model forward signatures unchanged

Complexity: O(n * h * d^2) naive outer-product is avoided; we implement normalized linear attention with
vector accumulators K and mixed outer S of shape [d x d]. Since v_dim == head_dim, per-step cost
is O(d * d) per head. If needed, this can be reduced further with low-rank, but remains sub-quadratic in seq len.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# -------------------------------
# Helper: Rotary Positional Embedding
# -------------------------------

def _build_rope_cache(seq_len: int, dim: int, device, base: float = 10000.0):
    half = dim // 2
    if half == 0:
        return None, None
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('t,d->td', t, inv_freq)  # [t, half]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos = rearrange(cos, 't d -> 1 t 1 d')
    sin = rearrange(sin, 't d -> 1 t 1 d')
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: Optional[torch.Tensor], sin: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Apply RoPE to tensor shaped [b, t, h, d]. If d is odd, last dim is left unchanged.
    cos/sin are shaped [1, t, 1, d//2]
    """
    if cos is None or sin is None:
        return x
    # Ensure dtype match for numerical stability and mixed precision compatibility
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    b, t, h, d = x.shape
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:half * 2]
    # Broadcast cos/sin: [1, t, 1, half]
    x1c = x1 * cos - x2 * sin
    x2c = x1 * sin + x2 * cos
    if d % 2 == 1:
        last = x[..., -1:]
        return torch.cat([x1c, x2c, last], dim=-1)
    else:
        return torch.cat([x1c, x2c], dim=-1)


# -------------------------------
# Kernel feature map
# -------------------------------

def _phi(x: torch.Tensor) -> torch.Tensor:
    # Positive feature map for kernelized softmax approximation
    return F.elu(x, alpha=1.0) + 1.0


# -------------------------------
# Core streaming kernel (compiled)
# -------------------------------

@torch.compile(mode="reduce-overhead", fullgraph=False)
def _stream_chunk(
    S: torch.Tensor,  # [b, h, M, d, d]
    K: torch.Tensor,  # [b, h, M, d]
    phi_q_chunk: torch.Tensor,  # [b, tc, h, d]
    phi_k_chunk: torch.Tensor,  # [b, tc, h, d]
    v_chunk: torch.Tensor,      # [b, tc, h, d]
    beta_chunk: torch.Tensor,   # [b, tc, h, M]
    mix_w_chunk: torch.Tensor,  # [b, tc, h, M]
    mask_chunk: Optional[torch.Tensor],  # [b, tc] or None
    eps: float
):
    b, tc, h, d = phi_q_chunk.shape
    M = S.shape[2]
    # Output tensor
    y_chunk = torch.empty((b, tc, h, d), device=phi_q_chunk.device, dtype=phi_q_chunk.dtype)

    # Iterate sequentially within chunk to preserve causality
    for t_idx in range(tc):
        phi_q_t = phi_q_chunk[:, t_idx]        # [b, h, d]
        phi_k_t = phi_k_chunk[:, t_idx]        # [b, h, d]
        v_t = v_chunk[:, t_idx]                # [b, h, d]
        beta_t = beta_chunk[:, t_idx]          # [b, h, M]
        mix_t = mix_w_chunk[:, t_idx]          # [b, h, M]

        if mask_chunk is not None:
            m_t = mask_chunk[:, t_idx]         # [b]
            m_t = rearrange(m_t, 'b -> b 1 1')  # [b, 1, 1]
            # Do not decay or update when masked: effective beta=1, updates=0
            beta_t = beta_t * m_t + (1.0 - m_t) * 1.0
            phi_k_t = phi_k_t * m_t
            v_t = v_t * m_t
            # mix_t can stay; output will be from previous state

        # Expand to timescales
        phi_k_tm = rearrange(phi_k_t, 'b h d -> b h 1 d').expand(-1, -1, M, -1)  # [b, h, M, d]
        v_tm = rearrange(v_t, 'b h d -> b h 1 d').expand(-1, -1, M, -1)          # [b, h, M, d]

        # Decay accumulators
        K = K * beta_t.unsqueeze(-1)  # [b, h, M, d]
        S = S * beta_t.unsqueeze(-1).unsqueeze(-1)  # [b, h, M, d, d]

        # Update accumulators: S += phi(k) ⊗ v; K += phi(k)
        # Outer product per timescale
        outer = torch.einsum('bhmd,bhmc->bhmdc', phi_k_tm, v_tm)  # [b, h, M, d, d]
        S = S + outer
        K = K + phi_k_tm

        # Compute output per timescale: y_m = (phi(q)^T S) / (phi(q)^T K + eps)
        phi_q_tm = rearrange(phi_q_t, 'b h d -> b h 1 d').expand(-1, -1, M, -1)  # [b, h, M, d]
        # Numerator: [b, h, M, d] = einsum over d: phi_q * S (last dim)
        num = torch.einsum('bhmd,bhmdc->bhmc', phi_q_tm, S)
        # Denominator: [b, h, M]
        den = torch.einsum('bhmd,bhmd->bhm', phi_q_tm, K) + eps
        y_m = num / den.unsqueeze(-1)

        # Mixture across timescales
        y = torch.sum(y_m * mix_t.unsqueeze(-1), dim=2)  # [b, h, d]
        y_chunk[:, t_idx] = y

    return y_chunk, S, K


# -------------------------------
# DeltaNet Attention Layer
# -------------------------------

class DeltaNet(nn.Module):
    """Single DeltaNet attention layer with normalized multi-timescale linear attention.

    forward signature preserved: (x: [batch, seq_len, hidden], mask: Optional[batch, seq_len])
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        )

        # Innovations
        self.num_timescales = kwargs.get('num_timescales', 2)
        self.beta_min = kwargs.get('beta_min', 0.85)
        self.beta_max = kwargs.get('beta_max', 0.9995)
        self.chunk_size = kwargs.get('chunk_size', 128)
        self.use_rope = kwargs.get('use_rope', True)
        self.rope_base = kwargs.get('rope_base', 10000.0)
        self.eps = kwargs.get('eps', 1e-6)

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Forgetting and mixture projections (content-dependent)
        self.beta_proj = nn.Linear(hidden_size, self.num_heads * self.num_timescales, bias=True)
        self.mix_proj = nn.Linear(hidden_size, self.num_heads * self.num_timescales, bias=True)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Initialize beta bias towards moderately high retention across timescales
        with torch.no_grad():
            # target retentions spaced between beta_min..beta_max
            if self.num_timescales > 1:
                targets = torch.linspace(self.beta_min, self.beta_max, steps=self.num_timescales)
            else:
                targets = torch.tensor([0.95])
            # Map to logits
            target_logits = torch.log(targets / (1 - targets))
            target_logits = repeat(target_logits, 'm -> (h m)', h=self.num_heads)
            self.beta_proj.bias.copy_(target_logits)
            nn.init.zeros_(self.mix_proj.weight)
            # Prefer long-ish timescales initially: uniform mixture
            nn.init.zeros_(self.mix_proj.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape

        residual = x

        # Project QKV from input
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # Optional RoPE on q and k for better positional inductive bias
        if self.use_rope:
            cos, sin = _build_rope_cache(t, self.head_dim, q.device, base=self.rope_base)
            # Cast to q/k dtype for mixed precision safety
            if cos is not None:
                cos = cos.to(dtype=q.dtype)
                sin = sin.to(dtype=q.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        # Feature map
        phi_q = _phi(q)
        phi_k = _phi(k)

        # Beta and mixture weights from content x
        beta_logits = self.beta_proj(x)  # [b, t, h*M]
        beta = rearrange(beta_logits, 'b t (h m) -> b t h m', h=self.num_heads, m=self.num_timescales)
        beta = torch.sigmoid(beta)
        # Clamp to stability range
        beta = beta.clamp(min=self.beta_min, max=self.beta_max)

        mix_logits = self.mix_proj(x)
        mix_w = rearrange(mix_logits, 'b t (h m) -> b t h m', h=self.num_heads, m=self.num_timescales)
        mix_w = F.softmax(mix_w, dim=-1)

        # Prepare accumulators per head and timescale
        S = torch.zeros((b, self.num_heads, self.num_timescales, self.head_dim, self.head_dim),
                        device=x.device, dtype=x.dtype)
        K = torch.zeros((b, self.num_heads, self.num_timescales, self.head_dim),
                        device=x.device, dtype=x.dtype)

        # Chunked streaming
        outputs = []
        for start in range(0, t, self.chunk_size):
            end = min(start + self.chunk_size, t)
            # Slice chunk
            phi_q_chunk = phi_q[:, start:end]
            phi_k_chunk = phi_k[:, start:end]
            v_chunk = v[:, start:end]
            beta_chunk = beta[:, start:end]
            mix_w_chunk = mix_w[:, start:end]
            mask_chunk = None
            if mask is not None:
                mask_chunk = mask[:, start:end].to(phi_q_chunk.dtype)

            y_chunk, S, K = _stream_chunk(
                S, K, phi_q_chunk, phi_k_chunk, v_chunk, beta_chunk, mix_w_chunk, mask_chunk, self.eps
            )
            outputs.append(y_chunk)

        y = torch.cat(outputs, dim=1)  # [b, t, h, d]
        y = rearrange(y, 'b t h d -> b t (h d)')

        # Output projection, dropout, residual, norm
        y = self.out_proj(y)
        y = self.dropout(y)
        y = self.layer_norm(residual + y)
        return y


# -------------------------------
# Transformer Block
# -------------------------------

class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block"""

    def __init__(self, hidden_size: int, num_heads: int = 8,
                 ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
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


# -------------------------------
# Model
# -------------------------------

class DeltaNetModel(nn.Module):
    """Complete DeltaNet model"""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1, **kwargs):
        """
        Accepts common alias keys via kwargs for compatibility with external training infra:
        - d_model -> hidden_size
        - n_heads -> num_heads
        - n_layers -> num_layers
        Extra kwargs are ignored to preserve interface.
        """
        # Map common aliases if provided
        if 'd_model' in kwargs and hidden_size == 512:
            hidden_size = kwargs.pop('d_model')
        if 'n_heads' in kwargs and num_heads == 8:
            num_heads = kwargs.pop('n_heads')
        if 'n_layers' in kwargs and num_layers == 6:
            num_layers = kwargs.pop('n_layers')
        # Ignore any other unknown kwargs
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
- Model Type: Normalized Linear Attention Transformer with Multi-Timescale Memory
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Innovations: normalized accumulators, multi-timescale forgetting, chunked streaming, RoPE
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
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4)

    batch_size, seq_len = 2, 129
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
