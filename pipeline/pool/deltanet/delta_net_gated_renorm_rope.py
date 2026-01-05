"""
DeltaNet: Gated Renormalized Linear Attention with Chunked Streaming
Evolved architecture implementing content-adaptive retention, S/Z normalization,
RoPE positional encoding, pre-LN RMSNorm, and SwiGLU FFN.
Maintains sub-quadratic complexity and strict causality with chunk-wise processing.
"""

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Safe torch.compile import
try:
    torch_compile = torch.compile  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    def torch_compile(fn=None, **kwargs):
        if fn is None:
            def wrapper(f):
                return f
            return wrapper
        return fn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (pre-LN style).
    Keeps mean-free scaling without centering, good for deep networks.
    """
    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        """Initialize RMSNorm layer.

        Args:
            dim: Dimension of the input features.
            eps: Small constant for numerical stability.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to input tensor.

        Args:
            x: Input tensor of shape [*, dim].

        Returns:
            Normalized tensor of same shape as input.
        """
        dim = x.shape[-1]
        rms = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms
        return x_norm * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split tensor in half along last dim and rotate for RoPE.

    Args:
        x: Input tensor of shape [..., d] where d is even.

    Returns:
        Rotated tensor of same shape with halves swapped and negated.
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to input tensor.

    Args:
        x: Input tensor of shape [b, h, t, d].
        cos: Cosine cache tensor of shape [1, 1, t, d].
        sin: Sine cache tensor of shape [1, 1, t, d].

    Returns:
        Tensor with RoPE applied, same shape as input.
    """
    return (x * cos) + (rotate_half(x) * sin)


def build_rope_cache(t: int, d: int, device: torch.device, base: float = 10000.0) -> tuple:
    """Build cosine and sine cache for Rotary Position Embedding.

    Args:
        t: Sequence length (number of positions).
        d: Head dimension (must be even).
        device: Torch device to create tensors on.
        base: Base frequency for computing inverse frequencies.

    Returns:
        Tuple of (cos, sin) tensors, each of shape [t, 1, d].
    """
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    positions = torch.arange(t, device=device, dtype=torch.float32)
    freqs = torch.einsum('t,d->td', positions, inv_freq)  # [t, half]
    cos = torch.cos(freqs).unsqueeze(1)  # [t, 1, half]
    sin = torch.sin(freqs).unsqueeze(1)  # [t, 1, half]
    cos = torch.stack([cos, cos], dim=-1).reshape(t, 1, d)
    sin = torch.stack([sin, sin], dim=-1).reshape(t, 1, d)
    return cos, sin


def phi_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Positive feature map for linear attention stability.

    Uses ELU + 1 as in Favor+ to ensure positivity.

    Args:
        x: Input tensor of shape [..., d].

    Returns:
        Positive feature tensor of same shape as input.
    """
    return F.elu(x, alpha=1.0) + 1.0


class DeltaNet(nn.Module):
    """Gated, Renormalized Linear Attention Layer with Chunked Streaming.

    Key features:
    - Content- and channel-wise retention r_t in [0,1] per head and per feature
    - S/Z renormalization accumulators for numerical stability
    - RoPE positional encoding on q/k
    - Chunk-wise causal processing
    - Batch-size agnostic, sub-quadratic complexity
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        """Initialize DeltaNet attention layer.

        Args:
            hidden_size: Total hidden dimension of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            **kwargs: Additional arguments, supports 'chunk_size' for chunked processing.

        Raises:
            AssertionError: If hidden_size is not divisible by num_heads.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Content-adaptive retention gate r_t in [0,1], per-head/per-channel
        # Implemented as sigmoid(W_g x + b), initialized to retain (~0.95)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Norms and dropout
        self.pre_norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Chunking and numerical stability
        self.chunk_size = kwargs.get('chunk_size', 128)
        self.eps = 1e-6

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize layer parameters with Xavier uniform and custom gate bias."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Gate bias to logit(0.95) ≈ 2.944439; weight small to start near constant retention
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.944439)

    @torch_compile()
    def _scan_chunks(self,
                     phi_q: torch.Tensor,
                     phi_k: torch.Tensor,
                     v: torch.Tensor,
                     r: torch.Tensor,
                     attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Perform causal chunked scan over time with S/Z renormalization.

        Args:
            phi_q: Query tensor after feature map, shape [b, h, t, d].
            phi_k: Key tensor after feature map, shape [b, h, t, d].
            v: Value tensor, shape [b, h, t, d].
            r: Retention gate tensor, shape [b, h, t, d].
            attn_mask: Optional mask [b, 1, t, 1], 1 for valid, 0 for pad.

        Returns:
            Output tensor of shape [b, h, t, d].
        """
        b, h, t, d = phi_q.shape
        y = torch.zeros((b, h, t, d), dtype=phi_q.dtype, device=phi_q.device)

        # Initialize states S and Z
        S = torch.zeros((b, h, d, d), dtype=phi_q.dtype, device=phi_q.device)
        Z = torch.zeros((b, h, d), dtype=phi_q.dtype, device=phi_q.device)

        # Process in chunks
        cs = self.chunk_size
        for start in range(0, t, cs):
            end = min(start + cs, t)
            # Slice views
            phi_q_c = phi_q[:, :, start:end, :]
            phi_k_c = phi_k[:, :, start:end, :]
            v_c = v[:, :, start:end, :]
            r_c = r[:, :, start:end, :]
            if attn_mask is not None:
                m_c = attn_mask[:, :, start:end, :]  # [b,1,c,1] broadcastable to [b,h,c,1]
            else:
                m_c = None

            # Step scan within chunk
            steps = end - start
            for i in range(steps):
                k_t = phi_k_c[:, :, i, :]  # [b,h,d]
                v_t = v_c[:, :, i, :]      # [b,h,d]
                q_t = phi_q_c[:, :, i, :]  # [b,h,d]
                r_t = r_c[:, :, i, :]      # [b,h,d] retention per channel

                if m_c is not None:
                    valid = m_c[:, :, i, :]  # [b,1,1]
                    # Broadcast to [b,h,1] for safe multiplication across feature dim
                    valid_bh1 = valid.expand(-1, h, -1)  # [b,h,1]
                    # Apply masking to k and v (zero out padded positions)
                    k_t = k_t * valid_bh1
                    v_t = v_t * valid_bh1
                    # For masked steps, set r_t=1 (no decay across padding)
                    r_t = torch.where(valid_bh1 > 0, r_t, torch.ones_like(r_t))

                # Update S and Z with retention r_t
                # S: [b,h,d,d] = r_t:[b,h,d] on row-dim
                S = S * r_t.unsqueeze(-1) + torch.einsum('bhd,bhe->bhde', k_t, v_t)
                Z = Z * r_t + k_t

                # Compute output y_t = (q_t @ S) / (q_t @ Z + eps)
                num_t = torch.einsum('bhd,bhde->bhe', q_t, S)  # [b,h,d]
                den_t = torch.sum(q_t * Z, dim=-1, keepdim=True)  # [b,h,1]
                den_t = den_t + self.eps
                y[:, :, start + i, :] = num_t / den_t

        return y

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the DeltaNet attention layer.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            mask: Optional attention mask of shape [batch, seq_len],
                where 1 indicates valid tokens and 0 indicates padding.

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size] with
            residual connection applied.
        """
        # Pre-norm
        x_norm = self.pre_norm(x)

        # Projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        g = self.gate_proj(x_norm)

        # Reshape to multi-head: [b, h, t, d]
        # Use einops rearrange to maintain robustness
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        g = rearrange(g, 'b t (h d) -> b h t d', h=self.num_heads)

        b, h, t, d = q.shape

        # Build RoPE cache and apply to q,k (requires even d)
        if d % 2 != 0:
            # If odd, skip RoPE to preserve robustness.
            pass
        else:
            cos, sin = build_rope_cache(t=t, d=d, device=x.device)
            cos = cos.to(dtype=q.dtype)
            sin = sin.to(dtype=q.dtype)
            # reshape to [1,1,t,d] for broadcasting across [b,h,t,d]
            cos = rearrange(cos, 't 1 d -> 1 1 t d')
            sin = rearrange(sin, 't 1 d -> 1 1 t d')
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Positive features for linear attention
        phi_q = phi_feature_map(q)
        phi_k = phi_feature_map(k)

        # Retention gate r_t in [0,1]
        r = torch.sigmoid(g)

        # Prepare mask for scanning: [b,1,t,1]
        if mask is not None:
            # Ensure mask dtype and device match compute tensor for multiplies
            m = mask.to(device=phi_q.device, dtype=phi_q.dtype)
            m = rearrange(m, 'b t -> b 1 t 1')
        else:
            m = None

        # Causal chunked scan
        y = self._scan_chunks(phi_q=phi_q, phi_k=phi_k, v=v, r=r, attn_mask=m)

        # Merge heads and output projection
        y = rearrange(y, 'b h t d -> b t (h d)')
        y = self.out_proj(y)
        y = self.dropout(y)

        # Residual connection (pre-LN style)
        return x + y


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network module."""

    def __init__(self, dim: int, hidden_mult: float = 2.0, dropout: float = 0.1):
        """Initialize SwiGLU FFN layer.

        Args:
            dim: Input and output dimension.
            hidden_mult: Multiplier for hidden layer size.
            dropout: Dropout probability.
        """
        super().__init__()
        inner = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.out = nn.Linear(inner, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            Transformed tensor of same shape as input.
        """
        a = self.w1(x)
        b = self.w2(x)
        x = F.silu(a) * b
        x = self.out(self.dropout(x))
        return x


class DeltaNetBlock(nn.Module):
    """Complete Transformer block with DeltaNet attention and SwiGLU FFN (pre-LN)."""

    def __init__(self, hidden_size: int, num_heads: int = 8,
                 ffn_hidden_size: Optional[int] = None, dropout: float = 0.1, **kwargs):
        """Initialize DeltaNet transformer block.

        Args:
            hidden_size: Hidden dimension of the model.
            num_heads: Number of attention heads.
            ffn_hidden_size: Hidden size for FFN. Defaults to 2x hidden_size.
            dropout: Dropout probability.
            **kwargs: Additional arguments passed to DeltaNet attention.
        """
        super().__init__()
        if ffn_hidden_size is None:
            # Use 2x SwiGLU instead of 4x GELU MLP for parameter efficiency
            ffn_hidden_size = int(hidden_size * 2.0)

        # Keep a norm here if needed in future extensions, but DeltaNet does its own pre-LN
        self.attn_norm = RMSNorm(hidden_size)
        self.attention = DeltaNet(hidden_size, num_heads, dropout, **kwargs)

        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, hidden_mult=2.0, dropout=dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            mask: Optional attention mask of shape [batch, seq_len].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        # Attention sublayer handles its own pre-LN and residual
        x = self.attention(x, mask)

        # Pre-LN FFN
        ffn_in = self.ffn_norm(x)
        x = x + self.ffn_dropout(self.ffn(ffn_in))
        return x


class DeltaNetModel(nn.Module):
    """Complete model using evolved DeltaNet attention blocks."""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
        num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1, **kwargs):
        """Initialize DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Hidden dimension of the model.
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads per block.
            max_seq_len: Maximum sequence length supported.
            dropout: Dropout probability.
            **kwargs: Additional arguments passed to DeltaNetBlock.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)

        # Stacked blocks
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads, dropout=dropout, **kwargs)
            for _ in range(num_layers)
        ])

        # Final normalization and LM head
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding parameters with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the full model.

        Args:
            input_ids: Token IDs of shape [batch, seq_len].
            attention_mask: Optional mask of shape [batch, seq_len],
                where 1 indicates valid tokens and 0 indicates padding.

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size].
        """
        b, t = input_ids.shape
        # positions
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        pos = pos.expand(b, -1)

        # embeddings
        tok = self.token_embedding(input_ids)
        pos = self.position_embedding(pos)
        x = tok + pos
        x = self.emb_dropout(x)

        # blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # head
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        """Generate a human-readable summary of the model architecture.

        Returns:
            Multi-line string describing model configuration and parameters.
        """
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Gated Renormalized Linear Attention Transformer
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Content-adaptive retention, S/Z renormalization, RoPE, Pre-LN RMSNorm, SwiGLU
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Factory function to create a DeltaNet language model.

    Args:
        vocab_size: Size of the vocabulary. Defaults to GPT-2 vocab size.
        **kwargs: Override default configuration (hidden_size, num_layers,
            num_heads, max_seq_len, dropout).

    Returns:
        Configured DeltaNetModel instance.
    """
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
    # Basic smoke test
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    with torch.no_grad():
        out = model(input_ids)
        print('Input:', input_ids.shape)
        print('Output:', out.shape)
        print(model.get_architecture_summary())
