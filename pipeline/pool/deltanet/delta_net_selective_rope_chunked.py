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
    """Root Mean Square Layer Normalization.

    A normalization layer that normalizes activations using only the root mean
    square of the values, without centering (no mean subtraction).

    Attributes:
        eps: Small constant for numerical stability.
        weight: Learnable scale parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initializes RMSNorm layer.

        Args:
            dim: Dimension of the input features to normalize.
            eps: Small constant added for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization to the input tensor.

        Args:
            x: Input tensor of shape [*, dim] where * means any number of
                leading dimensions.

        Returns:
            Normalized tensor of the same shape as input, scaled by learned weight.
        """
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def _rope_freqs(dim: int, seq_len: int, device, dtype, theta: float = 10000.0):
    """Computes rotary position embedding frequencies.

    Generates cosine and sine frequency tensors for rotary position embeddings.
    For even dimensions, pairs are rotated together. For odd dimensions, the
    last dimension remains unrotated.

    Args:
        dim: Dimension of the head (typically head_dim). Must be even for
            perfect pairing.
        seq_len: Sequence length to generate frequencies for.
        device: Device to place the tensors on.
        dtype: Data type for the tensors.
        theta: Base frequency for the rotary embeddings. Defaults to 10000.0.

    Returns:
        A tuple of (cos, sin) tensors, each of shape [seq_len, dim//2].
    """
    half = dim // 2
    if half == 0:
        cos = torch.ones(seq_len, 1, device=device, dtype=dtype)
        sin = torch.zeros(seq_len, 1, device=device, dtype=dtype)
        return cos, sin
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum('l,d->ld', t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embeddings to input tensor.

    Rotates pairs of dimensions in the last axis using precomputed cosine and
    sine frequencies. For odd-dimensional inputs, the last dimension remains
    unchanged.

    Args:
        x: Input tensor of shape [B, H, L, D] where B is batch size, H is
            number of heads, L is sequence length, and D is head dimension.
        cos: Cosine frequencies of shape [L, D//2].
        sin: Sine frequencies of shape [L, D//2].

    Returns:
        Tensor of the same shape as input with rotary embeddings applied.
    """
    B, H, L, D = x.shape
    half = D // 2
    if half == 0:
        return x
    x1 = x[..., :half]
    x2 = x[..., half:half*2]
    cos_ = cos.view(1, 1, L, half)
    sin_ = sin.view(1, 1, L, half)
    x1_ro = x1 * cos_ - x2 * sin_
    x2_ro = x1 * sin_ + x2 * cos_
    if D % 2 == 0:
        return torch.cat([x1_ro, x2_ro], dim=-1)
    else:
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
    """Performs vectorized token-selective delta rule with chunkwise scanning.

    Implements the core delta attention mechanism with token-conditional gating.
    The scan processes the sequence in chunks for memory efficiency while
    maintaining an associative memory matrix that is updated via outer products.
    Causality is inherent by the forward-only scan direction.

    Args:
        q: Query tensor of shape [B, H, L, Dh] where B is batch size, H is
            number of heads, L is sequence length, and Dh is head dimension.
        k: Key tensor of shape [B, H, L, Dh].
        v: Value tensor of shape [B, H, L, Dh].
        beta: Forgetting gate values in range [0, 1] of shape [B, H, L].
        in_gate: Input gate values in range [0, 1] of shape [B, H, L].
        out_gate: Output gate values in range [0, 1] of shape [B, H, L].
        attn_mask: Optional attention mask of shape [B, L] with 1 for valid
            tokens and 0 for padding tokens.
        chunk_size: Number of tokens to process in each chunk.

    Returns:
        Output tensor of shape [B, H, L, Dh] containing the attention results.
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
    """Single DeltaNet attention layer with token-conditional gating.

    Implements a linear attention mechanism with token-conditional forgetting,
    input, and output gates. Uses rotary position embeddings (RoPE) for Q/K
    and processes sequences in chunks for memory efficiency.

    Attributes:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        chunk_size: Number of tokens processed per chunk.
        rope_theta: Base frequency for rotary embeddings.
    """

    def __init__(self, hidden_size: Optional[int] = None, d_model: Optional[int] = None,
                 num_heads: int = 8, dropout: float = 0.1,
                 chunk_size: int = 128, rope_theta: float = 10000.0, **kwargs):
        """Initializes the DeltaNet attention layer.

        Args:
            hidden_size: Model hidden dimension. Either this or d_model must
                be specified.
            d_model: Alias for hidden_size for interface compatibility.
            num_heads: Number of attention heads. Defaults to 8.
            dropout: Dropout probability. Defaults to 0.1.
            chunk_size: Number of tokens to process per chunk. Defaults to 128.
            rope_theta: Base frequency for RoPE. Defaults to 10000.0.
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            ValueError: If neither hidden_size nor d_model is specified.
            AssertionError: If hidden_size is not divisible by num_heads.
        """
        super().__init__()
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNet requires hidden_size or d_model to be specified")
        if hidden_size is None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")

        self.chunk_size = int(chunk_size)
        self.rope_theta = float(rope_theta)

        self.pre_norm = RMSNorm(hidden_size)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.in_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.out_gate_proj = nn.Linear(hidden_size, num_heads, bias=True)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes layer parameters with appropriate strategies.

        Uses Xavier uniform for projection weights and initializes gate biases
        to favor retention (slow forgetting) while allowing information flow.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.beta_proj.bias, 2.0)
        nn.init.constant_(self.in_gate_proj.bias, 1.0)
        nn.init.constant_(self.out_gate_proj.bias, 1.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass through the DeltaNet attention layer.

        Args:
            x: Input tensor of shape [B, L, D] where B is batch size, L is
                sequence length, and D is hidden dimension.
            mask: Optional attention mask of shape [B, L] with 1 for valid
                tokens to attend and 0 for padding.

        Returns:
            Output tensor of shape [B, L, D] with residual connection applied.
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
    """Complete DeltaNet transformer block with attention and feed-forward.

    Combines DeltaNet attention with a feed-forward network (FFN) using
    pre-normalization and residual connections.

    Attributes:
        attention: DeltaNet attention layer.
        ffn: Feed-forward network with GELU activation.
        ffn_layer_norm: RMS normalization before FFN.
    """

    def __init__(self, hidden_size: Optional[int] = None, d_model: Optional[int] = None,
                 num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        """Initializes the DeltaNet transformer block.

        Args:
            hidden_size: Model hidden dimension. Either this or d_model must
                be specified.
            d_model: Alias for hidden_size for interface compatibility.
            num_heads: Number of attention heads. Defaults to 8.
            ffn_hidden_size: Hidden dimension of the FFN. Defaults to
                4 * hidden_size if not specified.
            dropout: Dropout probability. Defaults to 0.1.

        Raises:
            ValueError: If neither hidden_size nor d_model is specified.
        """
        super().__init__()
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNetBlock requires hidden_size or d_model to be specified")
        if hidden_size is None:
            hidden_size = d_model

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        self.attention = DeltaNet(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass through the transformer block.

        Args:
            x: Input tensor of shape [B, L, D].
            mask: Optional attention mask of shape [B, L].

        Returns:
            Output tensor of shape [B, L, D] after attention and FFN.
        """
        x = self.attention(x, mask)
        residual = x
        x = self.ffn(self.ffn_layer_norm(x))
        x = residual + x
        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet language model with embeddings and LM head.

    A full transformer-style language model using DeltaNet attention blocks.
    Includes token and position embeddings, stacked DeltaNet blocks, and a
    tied embedding LM head.

    Attributes:
        hidden_size: Model hidden dimension.
        max_seq_len: Maximum supported sequence length.
        token_embedding: Token embedding layer.
        position_embedding: Learnable position embedding layer.
        layers: Stack of DeltaNetBlock layers.
        layer_norm: Final layer normalization.
        lm_head: Language model output head (tied with token embeddings).
    """

    def __init__(self, vocab_size: int, hidden_size: Optional[int] = None, d_model: Optional[int] = None,
                 num_layers: int = 6, num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
        """Initializes the DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Model hidden dimension. Either this or d_model must
                be specified.
            d_model: Alias for hidden_size for interface compatibility.
            num_layers: Number of DeltaNet blocks. Defaults to 6.
            num_heads: Number of attention heads per block. Defaults to 8.
            max_seq_len: Maximum sequence length. Defaults to 2048.
            dropout: Dropout probability. Defaults to 0.1.

        Raises:
            ValueError: If neither hidden_size nor d_model is specified.
        """
        super().__init__()
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNetModel requires hidden_size or d_model to be specified")
        if hidden_size is None:
            hidden_size = d_model

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes embedding weights with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs forward pass through the complete model.

        Args:
            input_ids: Input token IDs of shape [B, L].
            attention_mask: Optional mask of shape [B, L] with 1 for valid
                tokens and 0 for padding.

        Returns:
            Logits tensor of shape [B, L, vocab_size].
        """
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
        """Returns a human-readable summary of the model architecture.

        Returns:
            A formatted string containing model type, dimensions, layer count,
            number of heads, max sequence length, key innovations, and total
            parameter count.
        """
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


def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Factory function for creating a DeltaNet model with default configuration.

    Creates a DeltaNetModel with sensible defaults that can be overridden via
    keyword arguments.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2 vocab).
        **kwargs: Configuration overrides. Supported keys include:
            - hidden_size (int): Model hidden dimension. Defaults to 512.
            - num_layers (int): Number of DeltaNet blocks. Defaults to 6.
            - num_heads (int): Number of attention heads. Defaults to 8.
            - max_seq_len (int): Maximum sequence length. Defaults to 2048.
            - dropout (float): Dropout probability. Defaults to 0.1.

    Returns:
        A configured DeltaNetModel instance.
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
