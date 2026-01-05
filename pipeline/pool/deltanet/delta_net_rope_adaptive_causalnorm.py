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
    """Apply positive feature map to stabilize causal normalization.

    Uses ELU activation shifted by 1 to ensure all outputs are positive,
    which is required for stable causal normalization.

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of the same shape with positive values (ELU(x) + 1).
    """
    return F.elu(x) + 1.0


def _build_rope_cache(seq_len: int, dim: int, device: torch.device, base: float = 10000.0):
    """Create RoPE cos/sin cache for given sequence length and head dimension.

    Builds precomputed cosine and sine values for Rotary Position Embeddings.
    Supports odd dimensions by rotating only the first 2*(dim//2) features.

    Args:
        seq_len: Sequence length for which to build the cache.
        dim: Head dimension (feature dimension per head).
        device: Device on which to create the tensors.
        base: Base value for computing inverse frequencies. Defaults to 10000.0.

    Returns:
        Tuple of (cos, sin) tensors, each of shape [seq_len, dim//2].
        For dim < 2, returns empty tensors to avoid NaN issues.
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
    """Apply rotary position embeddings to the last dimension of input tensor.

    Rotates pairs of features using precomputed cos/sin values to encode
    positional information. Supports odd dimensions by leaving the last
    feature unrotated.

    Args:
        x: Input tensor of shape [B, T, H, D] where B is batch size,
            T is sequence length, H is number of heads, D is head dimension.
        cos: Cosine values of shape [T, D//2] from RoPE cache.
        sin: Sine values of shape [T, D//2] from RoPE cache.

    Returns:
        Tensor of shape [B, T, H, D] with rotary position embeddings applied.
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
        """Decorator that applies torch.compile to a function for optimization.

        Args:
            fn: Function to be compiled with torch.compile.

        Returns:
            Compiled version of the function with dynamic shapes enabled.
        """
        return torch.compile(fn, fullgraph=False, dynamic=True)
else:
    def _compile_decorator(fn):
        """No-op decorator for environments without torch.compile.

        Args:
            fn: Function to return unchanged.

        Returns:
            The input function unchanged.
        """
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
    """Perform causal delta-rule scan over a chunk with causal normalization.

    Processes a chunk of the sequence using the delta rule with adaptive
    forget gates and causal normalization for stable linear attention.

    Args:
        q_chunk: Query tensor of shape [B, T, H, D].
        k_chunk: Key tensor of shape [B, T, H, D].
        v_chunk: Value tensor of shape [B, T, H, D].
        f_chunk: Forget gate values of shape [B, T, H], values in [0, 1].
        S: Associative state matrix of shape [B, H, D, D].
        m: Normalizer state of shape [B, H, D].
        eps: Small epsilon for numerical stability in normalization.
        mask_chunk: Optional attention mask of shape [B, T] with 1 for valid
            tokens and 0 for padding. Defaults to None.

    Returns:
        Tuple containing:
            - out_chunk: Output tensor of shape [B, T, H, D].
            - S: Updated associative state of shape [B, H, D, D].
            - m: Updated normalizer state of shape [B, H, D].
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
        """Initialize DeltaNet attention layer.

        Args:
            hidden_size: Total hidden dimension size.
            num_heads: Number of attention heads. Defaults to 8.
            dropout: Dropout probability. Defaults to 0.1.
            **kwargs: Additional keyword arguments:
                - chunk_size: Size of chunks for processing. Defaults to 256.
                - rope_base: Base for RoPE frequencies. Defaults to 10000.0.
                - eps: Epsilon for numerical stability. Defaults to 1e-6.

        Raises:
            AssertionError: If hidden_size is not divisible by num_heads.
        """
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
        """Initialize layer parameters.

        Uses Xavier uniform initialization for projection weights and zeros
        for output projection bias. The forget gate bias is initialized to
        target a relatively long half-life (τ0 ≈ 512 tokens).
        """
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
        """Perform forward pass through DeltaNet attention layer.

        Applies adaptive gating, RoPE, and causal normalization with chunkwise
        processing for efficient linear attention computation.

        Args:
            x: Input tensor of shape [B, T, C] where B is batch size,
                T is sequence length, C is hidden dimension.
            mask: Optional attention mask of shape [B, T] with 1 for valid
                tokens and 0 for padding. Defaults to None.

        Returns:
            Output tensor of shape [B, T, C] after attention, residual
            connection, and layer normalization.
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
        """Initialize DeltaNet transformer block.

        Args:
            hidden_size: Hidden dimension size for the model.
            num_heads: Number of attention heads. Defaults to 8.
            ffn_hidden_size: Hidden size for feed-forward network. Defaults
                to 4 * hidden_size if not specified.
            dropout: Dropout probability. Defaults to 0.1.
        """
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
        """Perform forward pass through transformer block.

        Applies DeltaNet attention followed by feed-forward network with
        residual connections and layer normalization.

        Args:
            x: Input tensor of shape [B, T, C].
            mask: Optional attention mask of shape [B, T]. Defaults to None.

        Returns:
            Output tensor of shape [B, T, C].
        """
        x = self.attention(x, mask)
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet model."""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
        """Initialize DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Hidden dimension size. Defaults to 512.
            num_layers: Number of transformer layers. Defaults to 6.
            num_heads: Number of attention heads. Defaults to 8.
            max_seq_len: Maximum sequence length. Defaults to 2048.
            dropout: Dropout probability. Defaults to 0.1.
        """
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
        """Initialize embedding parameters with normal distribution (std=0.02)."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform forward pass through the DeltaNet model.

        Args:
            input_ids: Token indices of shape [B, T] where B is batch size
                and T is sequence length.
            attention_mask: Optional mask of shape [B, T] with 1 for valid
                tokens and 0 for padding. Defaults to None.

        Returns:
            Logits tensor of shape [B, T, vocab_size].
        """
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
        """Generate a human-readable summary of the model architecture.

        Returns:
            Multi-line string describing model type, dimensions, layers,
            heads, key innovations, and parameter count.
        """
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
    """Factory function to create a DeltaNet model with default configuration.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2).
        **kwargs: Override default configuration options:
            - hidden_size: Hidden dimension. Defaults to 512.
            - num_layers: Number of layers. Defaults to 6.
            - num_heads: Number of attention heads. Defaults to 8.
            - max_seq_len: Maximum sequence length. Defaults to 2048.
            - dropout: Dropout probability. Defaults to 0.1.

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
