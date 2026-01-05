"""
DeltaNet: Normalized Linear Attention with Multi-Timescale Gating and RoPE
This evolved architecture upgrades the baseline DeltaNet with:
- Denominator-tracked linear attention using positive feature maps (elu + 1)
- Content-adaptive exponential forgetting (beta) with per-head log-timescale priors
- Input write gate per head
- Rotary positional embeddings (RoPE) for q/k
- Chunkwise causal processing with FP32 state accumulators
- Batch-size agnostic, einops-based reshaping throughout

Complexity: O(T * H * D^2) for head_dim D (sub-quadratic in sequence length)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Safe torch.compile decorator that degrades gracefully if unavailable
try:
    _torch_compile = torch.compile  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - environments without torch.compile
    def _torch_compile(fn=None, **kwargs):  # noqa: N802
        def wrapper(f):
            return f
        if fn is None:
            return wrapper
        return fn


@_torch_compile(dynamic=True)
def _delta_step(N, D, phi_q_t, phi_k_t, v_t, beta_t, g_t, eps: float):
    """Perform a single-timestep normalized linear attention update.

    This function is vectorized across batch and heads and performs the
    recurrent state update for normalized linear attention.

    Args:
        N: FP32 numerator state tensor of shape ``[B, H, D, E]``.
        D: FP32 denominator state tensor of shape ``[B, H, D]``.
        phi_q_t: Query feature map tensor of shape ``[B, H, D]``.
        phi_k_t: Key feature map tensor of shape ``[B, H, D]``.
        v_t: Value tensor of shape ``[B, H, E]``.
        beta_t: Exponential decay factor in ``(0, 1)`` of shape ``[B, H]``.
        g_t: Input write gate in ``(0, 1)`` of shape ``[B, H]``.
        eps: Small scalar for numerical stability in denominator.

    Returns:
        tuple: A tuple containing:
            - N: Updated numerator state ``[B, H, D, E]``.
            - D: Updated denominator state ``[B, H, D]``.
            - out_t: Output tensor ``[B, H, E]``.
    """
    # Expand beta and gate to match state shapes
    beta_N = beta_t[..., None, None]   # [B, H, 1, 1]
    beta_D = beta_t[..., None]         # [B, H, 1]
    g_N = g_t[..., None, None]         # [B, H, 1, 1]
    g_D = g_t[..., None]               # [B, H, 1]

    # FP32 on states and updates for numerical stability
    phi_k32 = phi_k_t.to(N.dtype)
    v32 = v_t.to(N.dtype)
    phi_q32 = phi_q_t.to(N.dtype)

    # Recurrent state updates (causal, normalized)
    # N_t = beta * N_{t-1} + g * (phi(k_t) ⊗ v_t)
    N = N * beta_N + torch.einsum('b h d, b h e -> b h d e', phi_k32, v32) * g_N
    # D_t = beta * D_{t-1} + g * phi(k_t)
    D = D * beta_D + phi_k32 * g_D

    # Output: y_t = (phi(q_t)^T N_t) / (eps + phi(q_t)^T D_t)
    # Numerator: [B, H, E]
    y_num = torch.einsum('b h d, b h d e -> b h e', phi_q32, N)
    # Denominator: [B, H]
    y_den = torch.einsum('b h d, b h d -> b h', phi_q32, D)
    out_t = y_num / (y_den[..., None] + eps)
    return N, D, out_t


@_torch_compile(dynamic=True)
def _process_chunk(N, D, phi_q_chunk, phi_k_chunk, v_chunk, beta_chunk, g_chunk, eps: float):
    """Process a chunk of timesteps using the recurrent delta step.

    This function iterates over a chunk of length L and applies the
    ``_delta_step`` function at each timestep in a compiled loop.

    Args:
        N: FP32 numerator state tensor of shape ``[B, H, D, E]``.
        D: FP32 denominator state tensor of shape ``[B, H, D]``.
        phi_q_chunk: Query feature map chunk of shape ``[B, H, L, D]``.
        phi_k_chunk: Key feature map chunk of shape ``[B, H, L, D]``.
        v_chunk: Value chunk of shape ``[B, H, L, D]``.
        beta_chunk: Decay factors for the chunk of shape ``[B, H, L]``.
        g_chunk: Gate values for the chunk of shape ``[B, H, L]``.
        eps: Small scalar for numerical stability.

    Returns:
        tuple: A tuple containing:
            - N: Updated numerator state ``[B, H, D, E]``.
            - D: Updated denominator state ``[B, H, D]``.
            - y_chunk: Output chunk ``[B, H, L, D]`` with same dtype as ``v_chunk``.
    """
    B, H, L, Dh = phi_q_chunk.shape
    # Preallocate output in the compute dtype of v_chunk to keep downstream dtype consistent
    y_chunk = torch.empty(B, H, L, v_chunk.shape[-1], device=v_chunk.device, dtype=v_chunk.dtype)
    for i in range(L):
        phi_q_t = phi_q_chunk[:, :, i, :]
        phi_k_t = phi_k_chunk[:, :, i, :]
        v_t = v_chunk[:, :, i, :]
        beta_t = beta_chunk[:, :, i]
        g_t = g_chunk[:, :, i]
        N, D, out_t = _delta_step(N, D, phi_q_t, phi_k_t, v_t, beta_t, g_t, eps)
        # Cast output back to v dtype for consistency with projection layers and residual
        y_chunk[:, :, i, :] = out_t.to(v_chunk.dtype)
    return N, D, y_chunk


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings (RoPE) to the input tensor.

    Rotates the input tensor's last dimension using precomputed cosine
    and sine frequency tensors to encode positional information.

    Args:
        x: Input tensor of shape ``[B, H, T, D]`` where D must be even.
        cos: Cosine frequencies of shape ``[T, D/2]``.
        sin: Sine frequencies of shape ``[T, D/2]``.

    Returns:
        torch.Tensor: Rotated tensor of shape ``[B, H, T, D]`` with
            positional embeddings applied.
    """
    # Split last dimension into even/odd pairs
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # Broadcast cos/sin to [1, 1, T, D/2]
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    # Rotate
    x_rot_even = x_even * cos_b - x_odd * sin_b
    x_rot_odd = x_even * sin_b + x_odd * cos_b
    # Interleave even/odd back to D using einops
    x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)
    x_rot = rearrange(x_rot, 'b h t d p -> b h t (d p)')
    return x_rot


class DeltaNet(nn.Module):
    """Normalized linear attention module with multi-timescale gating and RoPE.

    This module implements a DeltaNet attention mechanism featuring:
        - Denominator-tracked linear attention using positive feature maps
        - Content-adaptive exponential forgetting with per-head log-timescale priors
        - Input write gate per head
        - Rotary positional embeddings (RoPE) for queries and keys
        - Chunkwise causal processing with FP32 state accumulators

    Args:
        hidden_size: Total hidden dimension size.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout probability. Defaults to 0.1.
        **kwargs: Additional keyword arguments (unused).

    Attributes:
        hidden_size: The hidden dimension size.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (hidden_size // num_heads).
        eps: Numerical stability constant.
        chunk_size: Size of chunks for chunked processing.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, \
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Gates and decays (per-head)
        self.decay_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        # Per-head log-timescale prior (w0): initialize to log(ln(2)/H) with H log-spaced in [32, 4096]
        H_min, H_max = 32.0, 4096.0
        Hs = torch.logspace(math.log10(H_min), math.log10(H_max), steps=num_heads)
        lambdas = math.log(2.0) / Hs  # lambda = ln(2)/H
        self.decay_w0 = nn.Parameter(torch.log(lambdas))  # log lambda prior per head

        # Layer norm and dropout (post-attention, as in baseline)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Numerical constants
        self.eps = 1e-6
        self.chunk_size = 64  # chunked processing by default

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters with Xavier uniform and zero initialization."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.gate_proj.bias)
        # Encourage mild decay initially by biasing decay_proj toward zero (relies on decay_w0 prior)
        nn.init.zeros_(self.decay_proj.bias)

    def _build_rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Build the rotary positional embedding cache for a given sequence length.

        Args:
            seq_len: Length of the sequence.
            device: Device to create tensors on.
            dtype: Data type for the output tensors.

        Returns:
            tuple: A tuple containing:
                - cos: Cosine frequencies of shape ``[T, D/2]``.
                - sin: Sine frequencies of shape ``[T, D/2]``.
        """
        d = self.head_dim
        half = d // 2
        # Standard RoPE frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / float(half)))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum('t , d -> t d', t, inv_freq)  # [T, D/2]
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the forward pass of the DeltaNet attention layer.

        Args:
            x: Input tensor of shape ``[B, T, C]`` where B is batch size,
                T is sequence length, and C is the hidden dimension.
            mask: Optional padding mask of shape ``[B, T]`` where 1 indicates
                valid tokens and 0 indicates padding. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape ``[B, T, C]`` after attention,
                residual connection, and layer normalization.
        """
        # x: [B, T, C]
        B, T, C = x.shape
        residual = x

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads: [B, H, T, D]
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        # Rotary positional embeddings for q/k
        cos, sin = self._build_rope_cache(T, device=x.device, dtype=x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Positive feature maps for normalized linear attention
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0

        # Gates and decays (per head, per token)
        # gate: [B, H, T]
        gate = torch.sigmoid(rearrange(self.gate_proj(x), 'b t h -> b h t'))
        # decay rate r = softplus(w0 + W x), beta = exp(-r)
        decay_raw = rearrange(self.decay_proj(x), 'b t h -> b h t') + self.decay_w0[None, :, None]
        rate = F.softplus(decay_raw)  # positive
        rate = torch.clamp(rate, min=1e-4, max=10.0)
        beta = torch.exp(-rate)  # in (0,1)

        # Optional padding mask handling: mask shape [B, T], 1=keep, 0=pad
        if mask is not None:
            mask_bt = mask.to(x.dtype)
            mask_bht = rearrange(mask_bt, 'b t -> b 1 t')
            gate = gate * mask_bht  # no updates on padded positions
            # Keep state unchanged over pads (no decay): set beta=1 where mask=0
            beta = torch.where(mask_bht > 0, beta, torch.ones_like(beta))

        # Initialize FP32 states
        D = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=torch.float32)
        N = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)

        # Chunked causal scan (compiled per-chunk for performance)
        outputs = []
        eps = self.eps
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            N, D, y_chunk = _process_chunk(
                N,
                D,
                phi_q[:, :, start:end, :],
                phi_k[:, :, start:end, :],
                v[:, :, start:end, :],
                beta[:, :, start:end],
                gate[:, :, start:end],
                eps,
            )
            outputs.append(y_chunk)

        # Concatenate outputs over time: list of [B, H, L, D] -> [B, H, T, D]
        y = torch.cat(outputs, dim=2)

        # Merge heads back to hidden size: [B, T, C]
        y = rearrange(y, 'b h t d -> b t (h d)')
        y = self.out_proj(y)
        y = self.dropout(y)
        # Residual connection + LayerNorm
        y = self.layer_norm(residual + y)
        return y


class DeltaNetBlock(nn.Module):
    """A transformer block combining DeltaNet attention with a feed-forward network.

    This block consists of a DeltaNet attention layer followed by a two-layer
    feed-forward network with GELU activation, both with residual connections
    and layer normalization.

    Args:
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads. Defaults to 8.
        ffn_hidden_size: Hidden size of the feed-forward network. Defaults to
            4 * hidden_size if not specified.
        dropout: Dropout probability. Defaults to 0.1.

    Attributes:
        attention: The DeltaNet attention module.
        ffn: The feed-forward network.
        ffn_layer_norm: Layer normalization after FFN.
    """

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
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the forward pass through the transformer block.

        Args:
            x: Input tensor of shape ``[B, T, C]``.
            mask: Optional padding mask of shape ``[B, T]``. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape ``[B, T, C]``.
        """
        # Self-attention
        x = self.attention(x, mask)
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


class DeltaNetModel(nn.Module):
    """A complete DeltaNet language model with embeddings and output projection.

    This model consists of token and position embeddings, multiple DeltaNetBlock
    layers, and a language modeling head with tied embeddings.

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Hidden dimension size. Defaults to 512.
        num_layers: Number of transformer layers. Defaults to 6.
        num_heads: Number of attention heads per layer. Defaults to 8.
        max_seq_len: Maximum sequence length. Defaults to 2048.
        dropout: Dropout probability. Defaults to 0.1.

    Attributes:
        hidden_size: The hidden dimension size.
        max_seq_len: Maximum supported sequence length.
        token_embedding: Token embedding layer.
        position_embedding: Position embedding layer.
        layers: ModuleList of DeltaNetBlock layers.
        lm_head: Language modeling output projection (tied with token_embedding).
    """

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
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding weights with normal distribution (std=0.02)."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the forward pass of the language model.

        Args:
            input_ids: Input token IDs of shape ``[B, T]``.
            attention_mask: Optional attention mask of shape ``[B, T]`` where
                1 indicates valid tokens and 0 indicates padding. Defaults to None.

        Returns:
            torch.Tensor: Logits over vocabulary of shape ``[B, T, vocab_size]``.
        """
        B, T = input_ids.shape
        # Create position IDs
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
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
        """Generate a human-readable summary of the model architecture.

        Returns:
            str: A formatted string containing model configuration details
                including type, dimensions, layer count, and parameter count.
        """
        return f"""
DeltaNet Architecture Summary:
- Model Type: Normalized Linear Attention Transformer (RoPE + Multi-timescale Gating)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Denominator-tracked linear attention, per-head adaptive decays, RoPE
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Create a DeltaNetModel with default or custom configuration.

    Factory function that creates a DeltaNetModel instance with sensible
    defaults that can be overridden via keyword arguments.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2 vocab size).
        **kwargs: Optional overrides for model configuration. Supported keys:
            - hidden_size (int): Hidden dimension. Defaults to 512.
            - num_layers (int): Number of layers. Defaults to 6.
            - num_heads (int): Number of attention heads. Defaults to 8.
            - max_seq_len (int): Maximum sequence length. Defaults to 2048.
            - dropout (float): Dropout probability. Defaults to 0.1.

    Returns:
        DeltaNetModel: A configured DeltaNet language model instance.
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
    # Smoke test
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
