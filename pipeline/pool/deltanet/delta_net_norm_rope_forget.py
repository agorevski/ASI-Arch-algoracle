"""
DeltaNet: Enhanced Linear Attention Architecture with Normalized Delta Rule
- Implements content-normalized linear attention (Performer-style numerator/denominator)
- Per-head content-dependent forgetting with safe initialization and clamping
- Chunkwise causal processing with fp32 state accumulators
- RoPE positional encoding for Q/K before feature map
- Batch-size agnostic and uses einops.rearrange for all reshaping
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Safe torch.compile decorator (no-op if not available)
try:
    torch_compile = torch.compile  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    def torch_compile(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    Applies normalization over the last dimension only, with learned scale.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm layer.

        Args:
            dim: The dimension of the input tensor to normalize.
            eps: Small constant for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to input tensor.

        Args:
            x: Input tensor of shape [*, D] where * is any number of leading
                dimensions and D is the feature dimension.

        Returns:
            Normalized tensor of the same shape as input, scaled by learned
            parameters.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.scale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of a tensor by splitting and swapping halves.

    This is a helper function for applying rotary position embeddings (RoPE).
    It splits the last dimension in half and rotates by negating and swapping.

    Args:
        x: Input tensor of shape [..., D] where D is the dimension to rotate.

    Returns:
        Rotated tensor of the same shape where the first half is negated
        second half and vice versa.
    """
    d = x.shape[-1]
    d2 = (d // 2) * 2
    x1 = x[..., :d2]
    x2a, x2b = x1.chunk(2, dim=-1)
    rot = torch.cat((-x2b, x2a), dim=-1)
    if d2 < d:
        # append the untouched last dim if odd
        rot = torch.cat([rot, x[..., d2:]], dim=-1)
    return rot


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings (RoPE) to input tensor.

    Applies rotary embeddings using precomputed cosine and sine values.
    This enables position-aware attention without explicit position encodings.

    Args:
        x: Input tensor of shape [B, T, H, D] where B is batch size, T is
            sequence length, H is number of heads, and D is head dimension.
        cos: Precomputed cosine values of shape [1, T, 1, D].
        sin: Precomputed sine values of shape [1, T, 1, D].

    Returns:
        Tensor of shape [B, T, H, D] with rotary embeddings applied.
    """
    x1 = x[..., : (x.shape[-1] // 2) * 2]
    x_rot = x1 * cos[..., : x1.shape[-1]] + rotate_half(x1) * sin[..., : x1.shape[-1]]
    # Construct output by assignment to avoid extra concat work
    out = x.clone()
    out[..., :x1.shape[-1]] = x_rot
    return out


def build_rope_cache(T: int, D: int, device, dtype, base: float = 10000.0) -> (torch.Tensor, torch.Tensor):
    """Build cached cosine and sine values for rotary position embeddings.

    Precomputes the trigonometric values needed for RoPE to avoid redundant
    computation during forward passes.

    Args:
        T: Sequence length (number of positions).
        D: Head dimension (embedding dimension per head).
        device: The torch device to create tensors on.
        dtype: The torch dtype for the output tensors.
        base: Base value for computing inverse frequencies. Defaults to 10000.0.

    Returns:
        A tuple of (cos, sin) tensors, each of shape [1, T, 1, D], containing
        precomputed cosine and sine values for each position.
    """
    # Ensure even pairing length for angle computation
    half = (D // 2) * 2
    if half == 0:
        # degenerate case
        cos = torch.ones(1, T, 1, D, device=device, dtype=dtype)
        sin = torch.zeros(1, T, 1, D, device=device, dtype=dtype)
        return cos, sin
    inv_freq = 1.0 / (base ** (torch.arange(0, half, 2, device=device, dtype=torch.float32) / half))
    # positions
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.einsum('t,f->tf', t, inv_freq)  # [T, half/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, half]
    # compute trig functions in float32 for broad device support, then cast
    cos = emb.cos().unsqueeze(0).unsqueeze(2).to(dtype)  # [1, T, 1, half]
    sin = emb.sin().unsqueeze(0).unsqueeze(2).to(dtype)  # [1, T, 1, half]
    if half < D:
        pad = D - half
        cos = F.pad(cos, (0, pad))
        sin = F.pad(sin, (0, pad))
    return cos, sin


class DeltaNet(nn.Module):
    """DeltaNet attention layer with normalized linear attention and content-aware forgetting.

    - Uses phi(x) = elu(x) + 1 to ensure positivity for kernel features.
    - Maintains per-head numerator S (fp32) and denominator z (fp32) with decay beta.
    - Processes sequence in fixed-size chunks to control memory.
    - Applies RoPE to q/k before feature mapping for better length extrapolation.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        """Initialize DeltaNet attention layer.

        Args:
            hidden_size: Total hidden dimension of the model.
            num_heads: Number of attention heads. Defaults to 8.
            dropout: Dropout probability for regularization. Defaults to 0.1.
            **kwargs: Additional keyword arguments (unused, for compatibility).

        Raises:
            AssertionError: If hidden_size is not divisible by num_heads.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Normalization and dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = RMSNorm(hidden_size)

        # Parameters for forgetting
        self.beta_min = 0.80  # conservative lower bound to avoid over-forgetting
        self.beta_max = 0.999  # upper bound for stability
        self.eps = 1e-6
        self.chunk_size = 128  # default chunk length for sequential scan

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters with appropriate strategies.

        Uses Xavier uniform initialization for projection weights and
        initializes the forgetting gate bias to encourage high retention
        (approximately 0.95) at the start of training.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Initialize beta bias towards high retention ~0.95
        target = 0.95
        bias_val = math.log(target / (1 - target))
        nn.init.constant_(self.beta_proj.bias, bias_val)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feature map phi(x) = elu(x) + 1 for linear attention.

        This ensures positive features which is required for the kernel-based
        attention approximation to work correctly.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Transformed tensor with the same shape, guaranteed to be positive.
        """
        return F.elu(x) + 1.0

    @torch_compile(mode="default", dynamic=True)
    def _linear_attention_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Perform chunked causal normalized linear attention scan.

        Processes the sequence in fixed-size chunks while maintaining causal
        dependencies. Uses fp32 accumulators for numerical stability and applies
        content-dependent forgetting via the beta parameter.

        Args:
            q: Query tensor of shape [B, T, H, D].
            k: Key tensor of shape [B, T, H, D].
            v: Value tensor of shape [B, T, H, D].
            beta: Per-head forgetting gate of shape [B, T, H] with values in [0, 1].
            mask: Optional attention mask of shape [B, T] with 1 for valid tokens
                and 0 for padding.
            cos: Precomputed RoPE cosine cache of shape [1, T, 1, D].
            sin: Precomputed RoPE sine cache of shape [1, T, 1, D].

        Returns:
            Output tensor of shape [B, T, H, D] containing the attention results.
        """
        B, T, H, D = q.shape
        device = q.device
        dtype = q.dtype
        # fp32 accumulators
        S = torch.zeros((B, H, D, D), device=device, dtype=torch.float32)
        z = torch.zeros((B, H, D), device=device, dtype=torch.float32)

        # preallocate output to avoid Python-side list ops under torch.compile
        out = torch.empty((B, T, H, D), device=device, dtype=dtype)

        # process in chunks
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            q_c = q[:, start:end]  # [B, Tc, H, D]
            k_c = k[:, start:end]
            v_c = v[:, start:end]
            beta_c = beta[:, start:end]  # [B, Tc, H]
            cos_c = cos[:, start:end]
            sin_c = sin[:, start:end]
            if mask is not None:
                mask_c = mask[:, start:end]  # [B, Tc]
            else:
                mask_c = None

            # Apply RoPE to q/k for this chunk
            q_c = apply_rope(q_c, cos_c, sin_c)
            k_c = apply_rope(k_c, cos_c, sin_c)

            # Feature maps
            phi_q = self._feature_map(q_c)  # [B, Tc, H, D]
            phi_k = self._feature_map(k_c)  # [B, Tc, H, D]

            Tc = phi_q.shape[1]

            # Iterate within chunk sequentially for causality
            for t in range(Tc):
                phi_q_t = phi_q[:, t]           # [B, H, D]
                phi_k_t = phi_k[:, t]           # [B, H, D]
                v_t = v_c[:, t]                 # [B, H, D]
                beta_t = beta_c[:, t]           # [B, H]

                # Clamp beta to [beta_min, beta_max]
                beta_t = beta_t.clamp(self.beta_min, self.beta_max)

                # If mask provided, construct valid flags
                if mask_c is not None:
                    valid = mask_c[:, t].to(q.dtype)  # [B]
                else:
                    valid = torch.ones(B, device=device, dtype=q.dtype)

                # broadcast shapes (fp32 for state ops)
                beta_f32 = beta_t.to(torch.float32)
                beta_S = beta_f32.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
                beta_z = beta_f32.unsqueeze(-1)                # [B, H, 1]
                valid_f32 = valid.to(torch.float32)
                valid_S = valid_f32.view(B, 1, 1, 1)           # [B, 1, 1, 1]
                valid_z = valid_f32.view(B, 1, 1)              # [B, 1, 1]
                valid_y = valid.view(B, 1, 1).to(dtype)        # [B, 1, 1]

                # Decay previous states only for valid positions (skip decay on padding)
                decay_S = beta_S * valid_S + (1.0 - valid_S)   # [B, H, 1, 1]
                decay_z = beta_z * valid_z + (1.0 - valid_z)   # [B, H, 1]
                S = S * decay_S
                z = z * decay_z

                # Update with current kv (outer product), gated by validity
                # update = phi_k_t^T ⊗ v_t: [B, H, D, D]
                update = torch.einsum('bhd,bhe->bhde', phi_k_t.to(torch.float32), v_t.to(torch.float32))
                S = S + update * valid_S
                z = z + phi_k_t.to(torch.float32) * valid_z

                # Readout: y = (phi_q^T S) / (phi_q^T z + eps)
                num = torch.einsum('bhd,bhde->bhe', phi_q_t.to(torch.float32), S)  # [B, H, D]
                den = (phi_q_t.to(torch.float32) * z).sum(dim=-1, keepdim=True) + self.eps  # [B, H, 1]
                y_t = (num / den).to(dtype)  # [B, H, D]
                # Zero outputs for invalid positions
                y_t = y_t * valid_y

                # write to output tensor
                out[:, start + t] = y_t

        return out

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute forward pass of the DeltaNet attention layer.

        Applies normalized linear attention with content-dependent forgetting
        and rotary position embeddings. Includes residual connection.

        Args:
            x: Input tensor of shape [B, T, hidden_size] where B is batch size,
                T is sequence length, and hidden_size is the model dimension.
            mask: Optional attention mask of shape [B, T] with 1 for valid tokens
                and 0 for padding. Defaults to None.

        Returns:
            Output tensor of shape [B, T, hidden_size] after attention and
            residual connection.
        """
        B, T, _ = x.shape
        residual = x

        # PreNorm
        x = self.layer_norm(x)

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to heads using einops
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # Content-dependent forgetting parameter per head per token
        beta_logits = self.beta_proj(x)                    # [B, T, H]
        beta = torch.sigmoid(beta_logits)                  # [B, T, H]

        # Build RoPE caches
        cos, sin = build_rope_cache(T=T, D=self.head_dim, device=x.device, dtype=x.dtype)

        # Run chunked normalized linear attention scan
        y = self._linear_attention_scan(q, k, v, beta, mask, cos, sin)  # [B, T, H, D]

        # Merge heads
        y = rearrange(y, 'b t h d -> b t (h d)')

        # Output projection, dropout and residual
        y = self.out_proj(y)
        y = self.dropout(y)
        out = residual + y
        return out


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block with PreNorm RMSNorm and FFN."""

    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        """Initialize a DeltaNet transformer block.

        Args:
            hidden_size: Total hidden dimension of the model.
            num_heads: Number of attention heads. Defaults to 8.
            ffn_hidden_size: Hidden dimension of the feed-forward network.
                Defaults to 4 * hidden_size if not specified.
            dropout: Dropout probability for regularization. Defaults to 0.1.
        """
        super().__init__()
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size
        self.attn_norm = RMSNorm(hidden_size)
        self.attention = DeltaNet(hidden_size, num_heads, dropout)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute forward pass through the transformer block.

        Applies PreNorm architecture with attention followed by feed-forward
        network, each with residual connections.

        Args:
            x: Input tensor of shape [B, T, hidden_size].
            mask: Optional attention mask of shape [B, T] with 1 for valid tokens
                and 0 for padding. Defaults to None.

        Returns:
            Output tensor of shape [B, T, hidden_size].
        """
        # PreNorm -> Attention
        attn_in = self.attn_norm(x)
        x = x + self.attention(attn_in, mask)
        # PreNorm -> FFN
        ffn_in = self.ffn_norm(x)
        x = x + self.ffn(ffn_in)
        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet language model."""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6, num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
        """Initialize the DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Hidden dimension of the model. Defaults to 512.
            num_layers: Number of transformer blocks. Defaults to 6.
            num_heads: Number of attention heads per layer. Defaults to 8.
            max_seq_len: Maximum sequence length supported. Defaults to 2048.
            dropout: Dropout probability for regularization. Defaults to 0.1.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding parameters with normal distribution.

        Uses standard deviation of 0.02 for both token and position embeddings,
        following common transformer initialization practices.
        """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute forward pass of the language model.

        Processes input tokens through embeddings, transformer layers, and
        produces logits over the vocabulary.

        Args:
            input_ids: Token indices of shape [B, T] where B is batch size
                and T is sequence length.
            attention_mask: Optional mask of shape [B, T] with 1 for valid tokens
                and 0 for padding. Defaults to None.

        Returns:
            Logits tensor of shape [B, T, vocab_size] containing unnormalized
            scores for each vocabulary token at each position.
        """
        B, T = input_ids.shape
        # Positions
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(B, -1)
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        x = self.dropout(x)
        # Layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        """Generate a human-readable summary of the model architecture.

        Returns:
            A formatted string containing model type, dimensions, number of
            layers and heads, maximum sequence length, key innovations, and
            total parameter count.
        """
        return f"""
DeltaNet Architecture Summary:
- Model Type: Normalized Linear Attention Transformer (DeltaNet)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Innovations: Content-normalized linear attention, RoPE, content-dependent forgetting, chunked fp32 accumulators
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model

def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Factory function to create a DeltaNet language model with default config.

    Provides sensible defaults for model hyperparameters while allowing
    customization through keyword arguments.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2 vocab size).
        **kwargs: Override default configuration parameters. Supported keys:
            - hidden_size (int): Model dimension. Default 512.
            - num_layers (int): Number of transformer blocks. Default 6.
            - num_heads (int): Number of attention heads. Default 8.
            - max_seq_len (int): Maximum sequence length. Default 2048.
            - dropout (float): Dropout probability. Default 0.1.

    Returns:
        A configured DeltaNetModel instance.
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
    # Basic smoke test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    b, t = 2, 100
    input_ids = torch.randint(0, 1000, (b, t))
    attn_mask = torch.ones(b, t, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, attn_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
