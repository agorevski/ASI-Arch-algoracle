"""
DeltaNet: Evolved Linear Attention Architecture with Multi-Timescale Memory and RoPE
Improvements implemented based on experimental evidence:
- Multi-timescale state per head with content-dependent forgetting (gated alphas)
- RoPE (rotary position embedding) applied to Q/K for relative position awareness
- Normalized Hebbian updates via RMSNorm on K/V and state norm control
- Chunkwise causal scanning for sub-quadratic, streaming-compatible processing
- Batch-size agnostic, dynamic-shape operations using einops.rearrange
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Compute the inverse sigmoid (logit) function.

    Args:
        x: Input tensor with values in the range (0, 1).

    Returns:
        Tensor with the logit of the input values.
    """
    return torch.log(x / (1 - x))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs by their RMS value and applies a learned scale parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize RMSNorm layer.

        Args:
            dim: Dimension of the input features to normalize.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to the input.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            Normalized tensor with the same shape as input.
        """
        # RMS over last dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight


def apply_rope(q: torch.Tensor, k: torch.Tensor, base: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding (RoPE) to query and key tensors.

    RoPE encodes relative positional information by rotating pairs of dimensions
    in the query and key vectors based on their position in the sequence.

    Args:
        q: Query tensor of shape [B, T, H, D] where B is batch size, T is
            sequence length, H is number of heads, D is head dimension.
        k: Key tensor of shape [B, T, H, D].
        base: Base frequency for the rotary embeddings. Higher values result
            in slower frequency decay across dimensions.

    Returns:
        Tuple of (q_rot, k_rot), both with the same shape as inputs [B, T, H, D].
    """
    device = q.device
    dtype = q.dtype
    B, T, H, D = q.shape
    # Ensure even dimension for RoPE pairing
    D2 = (D // 2) * 2
    if D2 == 0:
        return q, k
    # Frequencies
    pos = torch.arange(T, device=device, dtype=dtype)  # [T]
    # Compute inverse frequency
    half_dim = D2 // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))  # [half_dim]
    # angles: [T, half_dim]
    angles = torch.einsum('t,d->td', pos, inv_freq)
    # cos/sin cached tensors: [1, T, 1, half_dim]
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]

    def rope_rotate(x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H, D]
        x_main = x[..., :D2]
        x_pass = x[..., D2:]
        x1 = x_main[..., :half_dim]
        x2 = x_main[..., half_dim:]
        # Apply rotation: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        x1c = x1 * cos - x2 * sin
        x2c = x1 * sin + x2 * cos
        out = torch.cat([x1c, x2c, x_pass], dim=-1)
        return out

    return rope_rotate(q), rope_rotate(k)


class DeltaNet(nn.Module):
    """
    Single DeltaNet attention layer with multi-timescale memory and RoPE.
    Preserves forward signature: forward(x: Tensor, mask: Optional[Tensor]) -> Tensor
    """

    def __init__(
        self,
        hidden_size: int = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        K: int = 4,  # number of timescales per head
        chunk_size: int = 64,
        rope_base: int = 10000,
        state_norm_clip: float = 2.0,  # clip threshold multiplier for state norm control
        alpha_min: float = 1e-4,
        alpha_max: float = 0.9995,
        d_model: Optional[int] = None,  # alias for hidden_size
        **kwargs,
    ):
        """Initialize DeltaNet attention layer.

        Args:
            hidden_size: Model dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
            dropout: Dropout probability applied to outputs.
            K: Number of timescales per head for multi-timescale memory.
            chunk_size: Size of chunks for chunked causal scanning.
            rope_base: Base frequency for rotary position embeddings.
            state_norm_clip: Clip threshold multiplier for state norm control.
            alpha_min: Minimum value for forgetting factor alpha.
            alpha_max: Maximum value for forgetting factor alpha.
            d_model: Alias for hidden_size (for compatibility).
            **kwargs: Additional arguments (ignored).

        Raises:
            ValueError: If neither hidden_size nor d_model is provided.
            AssertionError: If hidden_size is not divisible by num_heads.
        """
        super().__init__()
        # Support both hidden_size and d_model for compatibility
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNet: either hidden_size or d_model must be provided")
        if hidden_size is None:
            hidden_size = d_model

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        )
        self.K = K
        self.chunk_size = chunk_size
        self.rope_base = rope_base
        self.state_norm_clip = state_norm_clip
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Projections
        self.in_norm = RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Content-dependent forgetting parameters (alphas in (0,1)) per head-timescale
        self.alpha_proj = nn.Linear(hidden_size, num_heads * K, bias=True)
        # Mixing weights across timescales for readout (softmax over K)
        self.mix_proj = nn.Linear(hidden_size, num_heads * K, bias=True)
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Normalization on K,V channels
        self.k_norm = RMSNorm(self.head_dim)
        self.v_norm = RMSNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.resid_norm = nn.LayerNorm(hidden_size)

        # Initialize parameters
        self._reset_parameters()

    def _base_half_lives(self) -> torch.Tensor:
        """Compute base half-lives for multi-timescale forgetting.

        Creates a geometric progression of half-lives across timescales,
        e.g., [32, 128, 512, 2048] for K=4 with base=32 and ratio=4.

        Returns:
            Tensor of shape [K] containing half-life values in steps.
        """
        # Geometric progression of half-lives over timescales
        # e.g., ~[32, 128, 512, 2048] for K=4
        # Use torch here; later moved to device/dtype as needed
        base = 32.0
        ratio = 4.0
        return torch.tensor([base * (ratio ** i) for i in range(self.K)], dtype=torch.float32)

    def _reset_parameters(self):
        """Initialize model parameters.

        Uses Xavier uniform initialization for projection weights and sets
        alpha_proj bias based on target forgetting factors from half-lives.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Initialize alpha_proj bias to match target forgetting factors from half-lives
        # alpha ~ exp(-ln(2)/H) per step, bounded to (alpha_min, alpha_max)
        half_lives = self._base_half_lives()
        alphas = torch.exp(-math.log(2.0) / half_lives).clamp(self.alpha_min, self.alpha_max)
        # Repeat per head
        alphas = alphas.repeat(self.num_heads)
        with torch.no_grad():
            self.alpha_proj.weight.zero_()  # start near base alpha via bias only
            self.alpha_proj.bias.copy_(inverse_sigmoid(alphas))
            # Initialize mix_proj to uniform mixing (bias 0 -> softmax ~ uniform with small weights)
            self.mix_proj.weight.zero_()
            self.mix_proj.bias.zero_()

    @torch.compile(fullgraph=False)
    def _scan_chunks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        mix_w: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Perform causal multi-timescale delta-rule scan in chunks.

        Processes the sequence in chunks for sub-quadratic complexity while
        maintaining a multi-timescale memory state with content-dependent
        forgetting.

        Args:
            q: Query tensor of shape [B, T, H, D].
            k: Key tensor of shape [B, T, H, D] (normalized externally).
            v: Value tensor of shape [B, T, H, D] (normalized externally).
            alpha: Forgetting factors of shape [B, T, H, K] per token/head/timescale.
            mix_w: Mixing weight logits of shape [B, T, H, K] for combining timescales.
            attn_mask: Optional mask of shape [B, T] where 1 is valid, 0 is masked.

        Returns:
            Output tensor of shape [B, T, H, D].
        """
        B, T, H, D = q.shape
        K = self.K
        # FP32 accumulator for stability
        dtype_acc = torch.float32
        h_state = torch.zeros((B, H, K, D, D), device=q.device, dtype=dtype_acc)
        outputs = []
        c = math.sqrt(D) * self.state_norm_clip  # state norm clip threshold
        eps = 1e-6

        # Prepare mask per token
        if attn_mask is not None:
            # Expect [B, T], allow broadcast later
            attn_mask = attn_mask.to(q.dtype)

        # Chunked processing
        for start in range(0, T, self.chunk_size):
            end = min(T, start + self.chunk_size)
            # Slice current chunk
            q_c = q[:, start:end]  # [B, C, H, D]
            k_c = k[:, start:end]
            v_c = v[:, start:end]
            alpha_c = alpha[:, start:end]  # [B, C, H, K]
            mix_c = mix_w[:, start:end]    # [B, C, H, K]
            if attn_mask is not None:
                m_c = attn_mask[:, start:end]  # [B, C]
            else:
                m_c = None

            Cc = q_c.shape[1]
            # Step through tokens in chunk (kept small to remain efficient under compile)
            for t in range(Cc):
                q_t = q_c[:, t]           # [B, H, D]
                k_t = k_c[:, t]
                v_t = v_c[:, t]
                a_t = alpha_c[:, t]       # [B, H, K]
                mix_t = mix_c[:, t]       # [B, H, K]

                if m_c is not None:
                    m_t = m_c[:, t].view(B, 1, 1)  # [B,1,1]
                else:
                    m_t = None

                # Expand shapes for broadcasting
                # h_state: [B, H, K, D, D]
                # a_t -> [B, H, K, 1, 1]
                a_t_e = a_t.unsqueeze(-1).unsqueeze(-1).to(dtype_acc)
                if m_t is not None:
                    # Do not update when mask is 0; but keep forgetting to allow flushing state
                    # We model masked tokens as not writing new outer product; reading output as zeros.
                    write_scale = m_t.to(dtype_acc)  # [B,1,1]
                    write_scale = write_scale.unsqueeze(-1).unsqueeze(-1)  # [B,1,1,1,1]
                else:
                    write_scale = None

                # Outer product per head-timescale: [B, H, K, D, D]
                # k_t, v_t: [B, H, D] -> expand to K dimension
                k_t_e = k_t.unsqueeze(2).to(dtype_acc)
                v_t_e = v_t.unsqueeze(2).to(dtype_acc)
                outer = torch.einsum('bhkd,bhke->bhkde', k_t_e, v_t_e)

                # Update state
                h_state = h_state * a_t_e + (outer if write_scale is None else outer * write_scale)

                # State norm control (clip)
                # Frobenius norm per (B,H,K): [B,H,K,1,1]
                norm = torch.linalg.vector_norm(h_state, ord=2, dim=(-2, -1), keepdim=True)
                clip = torch.clamp(c / (norm + eps), max=1.0)
                h_state = h_state * clip

                # Readout: o_tk = einsum(q_t, h_state) -> [B, H, K, D]
                q_t_e = q_t.unsqueeze(2).to(dtype_acc)  # [B,H,1,D]
                o_tk = torch.einsum('bhkd,bhkde->bhke', q_t_e, h_state)
                # Mix across K with softmax weights
                w_t = F.softmax(mix_t, dim=-1).to(dtype_acc)  # [B,H,K]
                w_t = w_t / (w_t.sum(dim=-1, keepdim=True) + eps)
                w_t_e = w_t.unsqueeze(-1)  # [B,H,K,1]
                o_t = (o_tk * w_t_e).sum(dim=2)  # [B,H,D]

                if m_t is not None:
                    o_t = o_t * m_t  # zero out outputs for masked positions

                outputs.append(o_t.to(q.dtype))

        # outputs is list length T of [B,H,D]; need to assemble to [B,T,H,D]
        out = torch.stack(outputs, dim=1)  # [B,T,H,D]
        return out

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the DeltaNet attention layer.

        Applies multi-timescale linear attention with RoPE, content-dependent
        forgetting, and residual connection with layer normalization.

        Args:
            x: Input tensor of shape [B, T, hidden_size].
            mask: Optional attention mask of shape [B, T] where 1 indicates
                valid tokens and 0 indicates masked/padded positions.

        Returns:
            Output tensor of shape [B, T, hidden_size].
        """
        B, T, _ = x.shape
        residual = x

        x_norm = self.in_norm(x)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Reshape to [B, T, H, D]
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # Apply RoPE to q and k
        q, k = apply_rope(q, k, base=self.rope_base)

        # Normalize k and v (RMSNorm across channel D)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Compute alphas and mix weights from x_norm
        alpha_logits = self.alpha_proj(x_norm)  # [B, T, H*K]
        alpha_logits = rearrange(alpha_logits, 'b t (h k) -> b t h k', h=self.num_heads, k=self.K)
        # Sigmoid to (0,1), then clamp
        alpha = torch.sigmoid(alpha_logits).clamp(self.alpha_min, self.alpha_max)  # [B,T,H,K]

        mix_logits = self.mix_proj(x_norm)  # [B, T, H*K]
        mix_logits = rearrange(mix_logits, 'b t (h k) -> b t h k', h=self.num_heads, k=self.K)
        # scan
        y = self._scan_chunks(q, k, v, alpha, mix_logits, mask)  # [B, T, H, D]

        # Merge heads and project out
        y = rearrange(y, 'b t h d -> b t (h d)')
        y = self.out_proj(y)
        y = self.dropout(y)

        out = self.resid_norm(residual + y)
        return out


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block with attention and feed-forward network."""

    def __init__(
        self,
        hidden_size: int = None,
        num_heads: int = 8,
        ffn_hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        d_model: Optional[int] = None,  # alias for hidden_size
        **kwargs,
    ):
        """Initialize a DeltaNet transformer block.

        Args:
            hidden_size: Model dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
            ffn_hidden_size: Hidden dimension of the feed-forward network.
                Defaults to 4 * hidden_size.
            dropout: Dropout probability.
            d_model: Alias for hidden_size (for compatibility).
            **kwargs: Additional arguments passed to DeltaNet attention.

        Raises:
            ValueError: If neither hidden_size nor d_model is provided.
        """
        super().__init__()

        # Support both hidden_size and d_model for compatibility
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNetBlock: either hidden_size or d_model must be provided")
        if hidden_size is None:
            hidden_size = d_model

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        self.attention = DeltaNet(hidden_size, num_heads, dropout, **kwargs)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

        self.ffn_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [B, T, hidden_size].
            mask: Optional attention mask of shape [B, T].

        Returns:
            Output tensor of shape [B, T, hidden_size].
        """
        x = self.attention(x, mask)
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet language model with RoPE inside attention.

    Uses relative position embeddings (RoPE) instead of absolute positional
    embeddings for improved length generalization.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = None,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        d_model: Optional[int] = None,  # alias for hidden_size
        **kwargs,
    ):
        """Initialize the DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Model dimension (must be divisible by num_heads).
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads per layer.
            max_seq_len: Maximum sequence length (stored for reference).
            dropout: Dropout probability.
            d_model: Alias for hidden_size (for compatibility).
            **kwargs: Additional arguments passed to DeltaNetBlock.

        Raises:
            ValueError: If neither hidden_size nor d_model is provided.
        """
        super().__init__()

        # Support both hidden_size and d_model for compatibility
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNetModel: either hidden_size or d_model must be provided")
        if hidden_size is None:
            hidden_size = d_model

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        # Keep position_embedding attribute for backward compatibility, but it is unused
        self.position_embedding = None
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [DeltaNetBlock(hidden_size, num_heads, dropout=dropout, **kwargs) for _ in range(num_layers)]
        )

        # Output layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters with normal distribution for embeddings."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the complete DeltaNet model.

        Args:
            input_ids: Token indices of shape [B, T].
            attention_mask: Optional mask of shape [B, T] where 1 indicates
                valid tokens and 0 indicates masked/padded positions.

        Returns:
            Logits tensor of shape [B, T, vocab_size].
        """
        B, T = input_ids.shape
        # Embeddings
        x = self.token_embedding(input_ids)
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
            Multi-line string describing the model configuration and
            key architectural innovations.
        """
        return f"""
DeltaNet Architecture Summary:
- Model Type: Linear Attention Transformer (Delta-rule with multi-timescale memory)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Innovations: Multi-timescale forgetting, RoPE, RMSNorm K/V, state norm control, chunked scan
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model

def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Create a DeltaNet model with default parameters.

    Factory function that creates and returns a fully initialized DeltaNetModel
    with sensible defaults for all configuration options.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2 vocab size).
        **kwargs: Override any default configuration parameters. Accepts:
            - hidden_size or d_model: Model dimension (default: 512).
            - num_layers: Number of transformer blocks (default: 6).
            - num_heads: Number of attention heads (default: 8).
            - max_seq_len: Maximum sequence length (default: 2048).
            - dropout: Dropout probability (default: 0.1).
            - Additional DeltaNet-specific parameters.

    Returns:
        Initialized DeltaNetModel instance.
    """
    # Map d_model -> hidden_size if provided
    if 'hidden_size' not in kwargs and 'd_model' in kwargs:
        kwargs['hidden_size'] = kwargs.pop('d_model')

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
    # Basic test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=3, num_heads=8)
    B, T = 2, 100
    input_ids = torch.randint(0, 1000, (B, T))
    attn_mask = torch.ones(B, T, dtype=torch.float32)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
