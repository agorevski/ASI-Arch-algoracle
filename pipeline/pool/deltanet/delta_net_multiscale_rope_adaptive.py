"""
DeltaNet: Evolved Linear Attention with Multi-Timescale Adaptive Forgetting and RoPE
- Implements a content-adaptive, multi-timescale Delta rule in sub-quadratic time
- Adds Rotary Positional Embeddings (RoPE) for improved relative position handling
- Uses chunk-wise causal scanning for efficiency and memory robustness
- Pre-LN with RMSNorm and residual scaling for stability
- Replaces all .view/.reshape with einops.rearrange as mandated
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------
# Utility Normalization
# -----------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A normalization layer that normalizes by the root mean square of the input,
    without subtracting the mean. More efficient than LayerNorm.

    Args:
        dim: The dimension of the input features.
        eps: A small constant for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to the input tensor.

        Args:
            x: Input tensor of shape [*, dim].

        Returns:
            Normalized tensor of the same shape as input.
        """
        # x: [*, dim]
        norm_x = x.norm(keepdim=True, dim=-1)
        denom = norm_x / math.sqrt(x.shape[-1])
        out = x / (denom + self.eps)
        return out * self.weight

# -----------------------------
# Rotary Positional Embeddings
# -----------------------------
class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for relative position encoding.

    Implements rotary embeddings that encode relative positions through rotation
    matrices, enabling better length generalization.

    Args:
        dim: The dimension of the embedding (must be even).
        base: The base for computing inverse frequencies. Defaults to 10000.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _build_cache(self, seq_len: int, device, dtype):
        """Build cosine and sine cache for rotary embeddings.

        Args:
            seq_len: The sequence length to build the cache for.
            device: The device to place the tensors on.
            dtype: The data type for the output tensors.

        Returns:
            A tuple of (cos, sin) tensors, each of shape [seq_len, dim/2].
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [T, dim/2]
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin

    def forward(self, x: torch.Tensor):
        """Compute rotary embeddings for the input tensor.

        Args:
            x: Input tensor of shape [B, T, H, D] where B is batch size,
                T is sequence length, H is number of heads, D is head dimension.

        Returns:
            A tuple of (cos, sin) tensors, each of shape [1, T, 1, D/2],
            broadcast-ready for applying to query and key tensors.
        """
        # x expected: [B, T, H, D]
        B, T, H, D = x.shape
        cos, sin = self._build_cache(T, x.device, x.dtype)
        # reshape to broadcast across B, H -> [1, T, 1, D/2]
        cos = rearrange(cos, 't d -> 1 t 1 d')
        sin = rearrange(sin, 't d -> 1 t 1 d')
        return cos, sin

@torch.compile
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to the input tensor.

    Uses interleaved pairs (even/odd indices) as complex number pairs
    for rotation in the embedding space.

    Args:
        x: Input tensor of shape [B, T, H, D].
        cos: Cosine values of shape [1, T, 1, D/2].
        sin: Sine values of shape [1, T, 1, D/2].

    Returns:
        Rotated tensor of the same shape as input.
    """
    # Correct RoPE application using interleaved pairs (even/odd) as complex pairs
    # x: [B, T, H, D], cos/sin: [1, T, 1, D/2]
    # Reshape last dim into pairs: D = 2 * (D/2)
    x_pair = rearrange(x, 'b t h (d two) -> b t h d two', two=2)
    x_real = x_pair[..., 0]
    x_imag = x_pair[..., 1]
    rx_real = x_real * cos - x_imag * sin
    rx_imag = x_real * sin + x_imag * cos
    x_rot = torch.stack([rx_real, rx_imag], dim=-1)
    x_rot = rearrange(x_rot, 'b t h d two -> b t h (d two)')
    return x_rot

# -----------------------------
# Core DeltaNet Layer (MANDATORY name: DeltaNet)
# -----------------------------
class DeltaNet(nn.Module):
    """
    DeltaNet attention layer with content-adaptive multi-timescale forgetting and RoPE.
    - Keeps forward signature compatible: forward(x: [B,T,C], mask: Optional[[B,T]]) -> [B,T,C]
    - Sub-quadratic complexity via sequential scan (O(T) per head) and chunked processing
    - Uses RMSNorm and Pre-LN at the block level; this class handles projections and scan
    """
    def __init__(self, hidden_size: Optional[int] = None, num_heads: int = 8, dropout: float = 0.1,
                 d_model: Optional[int] = None, **kwargs):
        super().__init__()
        # Support both hidden_size and d_model aliases
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNet requires hidden_size or d_model to be specified")
        if hidden_size is None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even to apply RoPE"

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Multi-timescale parameters
        self.num_timescales = 2  # K, default enabled
        self.beta_proj = nn.Linear(hidden_size, self.num_heads * self.num_timescales, bias=True)
        self.mix_proj = nn.Linear(hidden_size, self.num_heads * self.num_timescales, bias=True)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Rotary embedding for Q/K
        self.rope = RotaryEmbedding(self.head_dim)

        # Scan parameters
        self.chunk_size = 128  # default chunk size for memory-efficient scanning

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters.

        Applies Xavier uniform initialization to projections and sets up
        initial biases for long retention and uniform timescale mixing.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Encourage long retention initially: sigmoid(4) ≈ 0.982
        nn.init.constant_(self.beta_proj.bias, 4.0)
        nn.init.zeros_(self.beta_proj.weight)
        # Start with uniform mixture over timescales by zero-init logits
        nn.init.zeros_(self.mix_proj.weight)
        nn.init.zeros_(self.mix_proj.bias)

    @torch.compile
    def _scan(self,
              q: torch.Tensor,  # [B, T, H, D]
              k: torch.Tensor,  # [B, T, H, D]
              v: torch.Tensor,  # [B, T, H, D]
              beta: torch.Tensor,  # [B, T, H, K]
              mix: torch.Tensor,   # [B, T, H, K]
              mask: torch.Tensor   # [B, T]
              ) -> torch.Tensor:
        """Perform causal sequential scan with multi-timescale memory.

        Iterates through the sequence maintaining per-timescale associative
        memory states, applying forgetting factors and mixing outputs.

        Args:
            q: Query tensor of shape [B, T, H, D].
            k: Key tensor of shape [B, T, H, D].
            v: Value tensor of shape [B, T, H, D].
            beta: Retention factors of shape [B, T, H, K] in range (0, 1).
            mix: Timescale mixing weights of shape [B, T, H, K], softmax normalized.
            mask: Attention mask of shape [B, T], 1 for valid tokens, 0 for padding.

        Returns:
            Output tensor of shape [B, T, H, D].
        """
        B, T, H, D = q.shape
        K = beta.shape[-1]
        # Initialize per-timescale associative memory: [B, H, K, D, D]
        S = torch.zeros((B, H, K, D, D), device=q.device, dtype=q.dtype)
        y_out = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)

        # Process in chunks to reduce overhead while keeping streamable semantics
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            for t in range(start, end):
                q_t = q[:, t]         # [B, H, D]
                k_t = k[:, t]         # [B, H, D]
                v_t = v[:, t]         # [B, H, D]
                beta_t = beta[:, t]   # [B, H, K]
                mix_t = mix[:, t]     # [B, H, K]
                m_t = mask[:, t].unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B,1,1,1,1]

                # Forgetting per timescale
                f_t = beta_t.unsqueeze(-1).unsqueeze(-1)  # [B, H, K, 1, 1]

                # Compute outer product once and broadcast across K
                # Base outer: [B, H, 1, D, D]
                outer_base = torch.matmul(k_t.unsqueeze(2).unsqueeze(-1), v_t.unsqueeze(2).unsqueeze(-2))
                # Expand across timescales K (no data copy)
                outer = outer_base.expand(-1, -1, K, -1, -1)  # [B, H, K, D, D]

                # Update associative memory with mask guarding
                S_prev = S
                S_new = f_t * S_prev + outer
                # If mask is 0 (padding), keep previous state
                S = torch.where(m_t > 0, S_new, S_prev)

                # Read: y_k = (q @ S)_k -> [B, H, K, D]
                # Explicitly expand q to match K to avoid ambiguous broadcasting in matmul
                q_exp = q_t.unsqueeze(2).unsqueeze(-2).expand(-1, -1, K, 1, -1)  # [B, H, K, 1, D]
                y_k = torch.matmul(q_exp, S).squeeze(-2)  # [B, H, K, D]

                # Mixture over timescales (softmax already applied outside)
                y_t = (y_k * mix_t.unsqueeze(-1)).sum(dim=2)  # [B, H, D]

                # If masked token, zero-out output
                y_t = torch.where(mask[:, t].unsqueeze(-1).unsqueeze(-1) > 0, y_t, torch.zeros_like(y_t))
                y_out[:, t] = y_t

        return y_out

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the DeltaNet attention layer.

        Args:
            x: Input tensor of shape [B, T, C] where B is batch size,
                T is sequence length, C is hidden size.
            mask: Optional attention mask of shape [B, T]. Values of 1 indicate
                valid tokens, 0 indicates padding. Defaults to all ones if not provided.

        Returns:
            Output tensor of shape [B, T, C].
        """
        B, T, C = x.shape
        if mask is None:
            mask = torch.ones((B, T), device=x.device, dtype=torch.int64)
        else:
            # Convert mask to int for where operations, expect 1 for valid and 0 for pad
            if mask.dtype != torch.int64 and mask.dtype != torch.int32:
                mask = (mask > 0).to(torch.int64)
            # Ensure mask is on the same device as x
            if mask.device != x.device:
                mask = mask.to(device=x.device)

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape to heads
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # RoPE for Q/K
        cos, sin = self.rope(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Multi-timescale parameters
        beta = self.beta_proj(x)  # [B, T, H*K]
        beta = rearrange(beta, 'b t (h k) -> b t h k', h=self.num_heads, k=self.num_timescales)
        beta = torch.sigmoid(beta)  # retention factors in (0,1)

        mix_logits = self.mix_proj(x)  # [B, T, H*K]
        mix_logits = rearrange(mix_logits, 'b t (h k) -> b t h k', h=self.num_heads, k=self.num_timescales)
        mix = torch.softmax(mix_logits, dim=-1)

        # Sequential scan with chunking (causal)
        y = self._scan(q, k, v, beta, mix, mask)  # [B, T, H, D]

        # Merge heads and project out
        y = rearrange(y, 'b t h d -> b t (h d)')
        y = self.out_proj(y)
        y = self.dropout(y)
        return y

# -----------------------------
# Transformer Block with Pre-LN and RMSNorm
# -----------------------------
class DeltaNetBlock(nn.Module):
    """Transformer block with Pre-LN, DeltaNet attention, and FFN.

    A complete transformer layer that combines RMSNorm pre-normalization,
    DeltaNet attention, and a feed-forward network with residual connections.

    Args:
        hidden_size: The hidden dimension size. Either this or d_model must be specified.
        num_heads: Number of attention heads. Defaults to 8.
        ffn_hidden_size: Hidden size of the feed-forward network.
            Defaults to 4 * hidden_size if not specified.
        dropout: Dropout probability. Defaults to 0.1.
        d_model: Alias for hidden_size for compatibility.
    """

    def __init__(self, hidden_size: Optional[int] = None, num_heads: int = 8,
                 ffn_hidden_size: Optional[int] = None, dropout: float = 0.1,
                 d_model: Optional[int] = None):
        super().__init__()
        # Support both hidden_size and d_model aliases
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNetBlock requires hidden_size or d_model to be specified")
        if hidden_size is None:
            hidden_size = d_model

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        # Pre-LN
        self.attn_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)

        self.attention = DeltaNet(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

        self.residual_scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [B, T, C].
            mask: Optional attention mask of shape [B, T].

        Returns:
            Output tensor of shape [B, T, C].
        """
        # Pre-LN Attention
        attn_in = self.attn_norm(x)
        attn_out = self.attention(attn_in, mask)
        x = x + self.residual_scale * attn_out

        # Pre-LN FFN
        ffn_in = self.ffn_norm(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.residual_scale * ffn_out
        return x

# -----------------------------
# Model Wrapper
# -----------------------------
class DeltaNetModel(nn.Module):
    """Complete DeltaNet language model with embeddings and output head.

    A full transformer language model using DeltaNet attention with
    multi-timescale adaptive forgetting and rotary positional embeddings.

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: The hidden dimension size. Defaults to 512.
        num_layers: Number of transformer layers. Defaults to 6.
        num_heads: Number of attention heads. Defaults to 8.
        max_seq_len: Maximum sequence length. Defaults to 2048.
        dropout: Dropout probability. Defaults to 0.1.
        d_model: Alias for hidden_size for compatibility.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1,
                 d_model: Optional[int] = None):
        super().__init__()
        # Support both hidden_size and d_model aliases
        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
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
            input_ids: Input token IDs of shape [B, T].
            attention_mask: Optional attention mask of shape [B, T].

        Returns:
            Logits tensor of shape [B, T, vocab_size].
        """
        B, T = input_ids.shape
        # Positions
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(B, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
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
            A formatted string describing the model configuration and parameter count.
        """
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Linear Attention Transformer with Multi-Timescale Delta Memory
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Innovations: RoPE, content-adaptive multi-timescale forgetting, Pre-LN RMSNorm, chunked scan
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""

# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Factory function to create a DeltaNetModel with default configuration.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2 vocab size).
        **kwargs: Additional keyword arguments to override default configuration.
            Supported keys: hidden_size, num_layers, num_heads, max_seq_len, dropout.

    Returns:
        An initialized DeltaNetModel instance.
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
    # Quick smoke test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    B, T = 2, 64
    input_ids = torch.randint(0, 1000, (B, T))
    attn_mask = torch.ones(B, T, dtype=torch.int64)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
