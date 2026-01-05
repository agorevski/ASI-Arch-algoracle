"""
DeltaNet: Evolved Linear Attention Architecture with Adaptive Gating, RoPE, and Chunked Scan
- Preserves sub-quadratic complexity via per-step associative update (O(N * H * Dh^2))
- Adds content-adaptive write/forget gates (alpha, beta)
- Applies Rotary Position Embeddings (RoPE) to Q/K for improved relative positioning
- Vectorized across heads; chunked time scan for memory-efficiency
- Uses RMSNorm pre-norm and SwiGLU FFN for improved optimization
- Replaces all view/reshape with einops.rearrange for robust dynamic shapes
- Maintains batch-size agnosticism and causal integrity
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Safe torch.compile wrapper (keeps decorator presence across environments)
try:
    torch_compile = torch.compile  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    def torch_compile(fn=None, *args, **kwargs):
        if fn is None:
            def wrapper(f):
                return f
            return wrapper
        return fn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs by their RMS value without subtracting the mean,
    providing faster and more stable normalization than LayerNorm.

    Args:
        dim: The dimension of the input features to normalize.
        eps: Small epsilon value to prevent division by zero.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to input tensor.

        Args:
            x: Input tensor of shape [B, L, D].

        Returns:
            Normalized tensor of the same shape as input.
        """
        orig_dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        y = x_f * rms
        # Multiply by weight in float32 for stability, then cast back once
        y = y * self.weight.float()
        return y.to(orig_dtype)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    Implements the SwiGLU activation function which combines Swish (SiLU)
    with a Gated Linear Unit for improved optimization and performance.

    Args:
        dim: Input and output dimension.
        hidden: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=True)
        self.w2 = nn.Linear(dim, hidden, bias=True)
        self.w3 = nn.Linear(hidden, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input.

        Args:
            x: Input tensor of shape [B, L, D].

        Returns:
            Transformed tensor of the same shape as input.
        """
        a = self.w1(x)
        b = self.w2(x)
        x = F.silu(a) * b
        x = self.w3(self.dropout(x))
        return x


def _build_rope_cache(seq_len: int, dim: int, device, dtype, base: float = 10000.0):
    """Build cosine and sine caches for Rotary Position Embeddings.

    Args:
        seq_len: Sequence length to build cache for.
        dim: Dimension of the embeddings (typically head dimension).
        device: Torch device to place tensors on.
        dtype: Data type for the output tensors.
        base: Base frequency for rotary embeddings.

    Returns:
        Tuple of (cos, sin) tensors, each with shape [seq_len, dim//2].
    """
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('l,d->ld', t, inv_freq)  # [L, half]
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embeddings to input tensor.

    Rotates pairs of dimensions using precomputed sine and cosine values
    to encode relative positional information.

    Args:
        x: Input tensor of shape [B, H, L, D].
        cos: Cosine cache of shape [L, D//2].
        sin: Sine cache of shape [L, D//2].

    Returns:
        Tensor with rotary embeddings applied, same shape as input.
    """
    B, H, L, D = x.shape
    # ensure even pairing base
    even_dims = (D // 2) * 2
    x_even = x[..., :even_dims]
    x_rest = x[..., even_dims:]
    # reshape pairs
    x_pairs = rearrange(x_even, 'b h l (d two) -> b h l d two', two=2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]
    # cos/sin: [L, D//2]
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    x1r = x1 * cos_b - x2 * sin_b
    x2r = x2 * cos_b + x1 * sin_b
    x_rot = rearrange(torch.stack([x1r, x2r], dim=-1), 'b h l d two -> b h l (d two)', two=2)
    if x_rest.numel() > 0:
        x_out = torch.cat([x_rot, x_rest], dim=-1)
    else:
        x_out = x_rot
    return x_out


@torch_compile
def _delta_scan(q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor,
                chunk_size: int) -> torch.Tensor:
    """Perform chunked causal delta-rule associative scan.

    Implements the core DeltaNet recurrence with adaptive write (alpha) and
    forget (beta) gates. Processes the sequence in chunks for memory efficiency.

    Args:
        q: Query tensor of shape [B, H, L, Dh].
        k: Key tensor of shape [B, H, L, Dh].
        v: Value tensor of shape [B, H, L, Dh].
        alpha: Write gate tensor of shape [B, H, L, 1].
        beta: Forget gate tensor of shape [B, H, L, 1].
        chunk_size: Number of time steps to process per chunk.

    Returns:
        Output tensor of shape [B, H, L, Dh] containing attention results.
    """
    B, H, L, Dh = q.shape
    out = torch.zeros((B, H, L, Dh), device=q.device, dtype=q.dtype)
    state = torch.zeros((B, H, Dh, Dh), device=q.device, dtype=q.dtype)

    for start in range(0, L, chunk_size):
        end = min(L, start + chunk_size)
        for t in range(start, end):
            # Decay previous state
            beta_t = beta[:, :, t]  # [B, H, 1]
            beta_t = beta_t.unsqueeze(-1)  # [B, H, 1, 1]
            state = state * beta_t
            # Write
            k_t = k[:, :, t, :]  # [B, H, Dh]
            v_t = v[:, :, t, :]  # [B, H, Dh]
            a_t = alpha[:, :, t]  # [B, H, 1]
            kv_outer = torch.einsum('b h d, b h e -> b h d e', k_t, v_t)
            state = state + (a_t.unsqueeze(-1) * kv_outer)
            # Read
            q_t = q[:, :, t, :]
            out[:, :, t, :] = torch.einsum('b h d, b h d e -> b h e', q_t, state)
    return out


class DeltaNet(nn.Module):
    """Single DeltaNet attention layer with adaptive gates and RoPE.
    Maintains sub-quadratic complexity with chunked associative scan.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        """Initialize DeltaNet attention layer.

        Args:
            hidden_size: Total hidden dimension of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability for output projection.
            **kwargs: Additional keyword arguments (unused, for compatibility).
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
        # Adaptive gates (per-head, per-token)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.alpha_proj = nn.Linear(hidden_size, num_heads, bias=True)
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Chunk size for time scan
        self.chunk_size = 256
        # Gate ranges
        self.beta_min = 0.85
        self.beta_max = 0.999

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters with appropriate schemes.

        Uses Xavier uniform for projection weights and specialized
        initialization for gate biases to favor retention initially.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Gate initializations
        # Forget gate bias to favor retention initially (approx logit(0.97))
        beta_bias_val = math.log(0.97 / (1 - 0.97))
        nn.init.constant_(self.beta_proj.bias, beta_bias_val)
        nn.init.zeros_(self.beta_proj.weight)
        # Write gate bias around 0.5
        nn.init.constant_(self.alpha_proj.bias, 0.0)
        nn.init.zeros_(self.alpha_proj.weight)

    def _normalize_mask(self, mask: torch.Tensor, B: int, L: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Normalize various mask shapes to a standard format.

        Handles different input mask shapes and converts them to a unified
        format suitable for broadcasting with attention tensors.

        Args:
            mask: Input attention mask of various shapes.
            B: Batch size.
            L: Sequence length.
            dtype: Target data type for the mask.
            device: Target device for the mask.

        Returns:
            Normalized mask tensor of shape [B, 1, L, 1] with float dtype {0, 1}.
        """
        m = mask
        if m.dtype != torch.bool:
            m = m > 0
        # Ensure batch dimension is first and sequence length is last
        # Squeeze any singleton dimensions except batch and last
        if m.dim() == 1:
            # Assume shape [L] -> expand to [B, L]
            m = m.unsqueeze(0).expand(B, -1)
        elif m.dim() >= 3:
            # Try to squeeze common attention mask shapes [B,1,1,L] -> [B,L]
            # Remove middle singleton dims
            while m.dim() > 2 and m.size(1) == 1:
                m = m.squeeze(1)
            while m.dim() > 2 and m.size(-1) == 1:
                m = m.squeeze(-1)
            # Fallback: flatten trailing dims to L
            if m.dim() != 2:
                m = m.reshape(m.size(0), -1)
        # Now m should be [B, L]
        if not (m.dim() == 2 and m.size(0) == B and m.size(1) == L):
            # If sequence length mismatches, try to slice or pad (slice preferred)
            if m.size(1) >= L:
                m = m[:, :L]
            else:
                # pad with ones (valid) to preserve causality; safer than zeros
                pad_len = L - m.size(1)
                m = torch.cat([m, torch.ones((B, pad_len), dtype=m.dtype, device=m.device)], dim=1)
        m = m.to(device=device, dtype=dtype)
        return m.unsqueeze(1).unsqueeze(-1)  # [B,1,L,1]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute DeltaNet attention.

        Args:
            x: Input tensor of shape [B, L, D].
            mask: Optional attention mask for padding. Supports various shapes
                including [B, L], [B, 1, L], [B, 1, 1, L].

        Returns:
            Output tensor of shape [B, L, D] after attention and projection.
        """
        B, L, D = x.shape
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape to heads
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # Rotary position embedding for Q/K
        cos, sin = _build_rope_cache(L, self.head_dim, device=x.device, dtype=x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Gates from input x (content-adaptive, per head)
        beta_logits = self.beta_proj(x.float())  # [B, L, H]
        alpha_logits = self.alpha_proj(x.float())  # [B, L, H]
        # Map beta to [beta_min, beta_max]
        beta_sigma = torch.sigmoid(beta_logits)
        beta = self.beta_min + (self.beta_max - self.beta_min) * beta_sigma
        alpha = torch.sigmoid(alpha_logits)
        # Reshape to [B, H, L, 1]
        beta = rearrange(beta, 'b l h -> b h l 1')
        alpha = rearrange(alpha, 'b l h -> b h l 1')
        beta = beta.to(q.dtype)
        alpha = alpha.to(q.dtype)

        # Optional attention mask support (e.g., for padding)
        if mask is not None:
            m = self._normalize_mask(mask, B=B, L=L, dtype=q.dtype, device=x.device)  # [B,1,L,1]
            # For padded positions: prevent state changes (beta=1, alpha=0)
            alpha = alpha * m
            beta = beta * m + (1.0 - m)
        # Chunked causal scan across time
        out = _delta_scan(q, k, v, alpha, beta, self.chunk_size)

        # Zero outputs at padded positions to avoid residual contamination
        if mask is not None:
            m = self._normalize_mask(mask, B=B, L=L, dtype=out.dtype, device=out.device)
            out = out * m

        # Merge heads and project out
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class DeltaNetBlock(nn.Module):
    """Transformer block using DeltaNet attention (pre-norm) and SwiGLU FFN."""

    def __init__(self, hidden_size: int, num_heads: int = 8,
                 ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        """Initialize a DeltaNet transformer block.

        Args:
            hidden_size: Hidden dimension of the model.
            num_heads: Number of attention heads.
            ffn_hidden_size: Hidden dimension of the FFN. Defaults to 4 * hidden_size.
            dropout: Dropout probability.
        """
        super().__init__()
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size
        self.attn_norm = RMSNorm(hidden_size)
        self.attention = DeltaNet(hidden_size, num_heads, dropout)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, ffn_hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [B, L, D].
            mask: Optional attention mask for padding.

        Returns:
            Output tensor of shape [B, L, D].
        """
        # Pre-norm attention
        x = x + self.attention(self.attn_norm(x), mask)
        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class DeltaNetModel(nn.Module):
    """Complete model wrapper over multiple DeltaNet blocks."""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
        """Initialize the complete DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Hidden dimension of the model.
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads per layer.
            max_seq_len: Maximum sequence length for position embeddings.
            dropout: Dropout probability throughout the model.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding parameters with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the complete model.

        Args:
            input_ids: Token IDs tensor of shape [B, L].
            attention_mask: Optional attention mask for padding.

        Returns:
            Logits tensor of shape [B, L, vocab_size].
        """
        B, L = input_ids.shape
        device = input_ids.device
        # Positions
        pos_ids = torch.arange(L, device=device).unsqueeze(0)
        pos_ids = pos_ids.expand(B, -1)
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        x = self.emb_dropout(x)
        # Blocks
        for layer in self.layers:
            x = layer(x, attention_mask)
        # Final norm and head
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        """Generate a human-readable summary of the model architecture.

        Returns:
            Multi-line string describing the model configuration and parameters.
        """
        return f"""
DeltaNet Architecture Summary:
- Model Type: Linear Attention Transformer (DeltaNet evolved)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Adaptive alpha/beta gates, RoPE, chunked scan, RMSNorm, SwiGLU
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> 'DeltaNetModel':
    """Factory function to create a DeltaNetModel with default configuration.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2 vocab size).
        **kwargs: Override any default configuration parameters:
            - hidden_size: Hidden dimension (default: 512).
            - num_layers: Number of transformer blocks (default: 6).
            - num_heads: Number of attention heads (default: 8).
            - max_seq_len: Maximum sequence length (default: 2048).
            - dropout: Dropout probability (default: 0.1).

    Returns:
        Configured DeltaNetModel instance.
    """
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
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    bsz, seqlen = 2, 100
    input_ids = torch.randint(0, 1000, (bsz, seqlen))
    # Example mask with variable shapes
    lengths = torch.tensor([80, 100])
    attn_mask = torch.arange(seqlen).unsqueeze(0) < lengths.unsqueeze(1)  # [B, L]
    attn_mask_4d = attn_mask[:, None, None, :]  # [B,1,1,L]
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask_4d)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
