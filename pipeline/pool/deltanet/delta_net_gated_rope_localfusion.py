"""
DeltaNet: Gated Linear Attention with RoPE and Local Window Fusion
Evolved architecture implementing:
- Gated multi-timescale delta memory (per-head, per-channel write + per-dim forget)
- Rotary Positional Embeddings (RoPE) on Q/K
- Chunked scanning for recurrent core, sub-quadratic O(N D^2)
- Chunked causal local attention with fixed window W for sharper token selectivity (O(N W))
- Head-parallel processing and einops for safe reshaping
- Optional attention mask support (batch-agnostic)

This implementation maintains interface compatibility and default-enable new features.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# -----------------------------
# Utility: RMSNorm (for stable pre-norm)
# -----------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for stable pre-normalization.

    Normalizes inputs by their RMS value and applies a learned scale.

    Attributes:
        eps: Small constant for numerical stability.
        weight: Learned per-dimension scale parameter.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        """Initializes RMSNorm.

        Args:
            d: Dimension of the input features.
            eps: Small constant added to denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization to the input tensor.

        Args:
            x: Input tensor of shape [..., d] where d is the feature dimension.

        Returns:
            Normalized tensor of the same shape as input, scaled by learned weights.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_norm = x * rms
        return x_norm * self.weight

# -----------------------------
# Rotary Positional Embedding helpers (no persistent buffers)
# -----------------------------
@torch.compile
def apply_rope(x_q: torch.Tensor, x_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Positional Embedding (RoPE) to queries and keys.

    Rotates the query and key vectors using precomputed cosine and sine
    values to encode positional information directly into the attention.

    Args:
        x_q: Query tensor of shape [b, t, h, d] where b is batch size,
            t is sequence length, h is number of heads, d is head dimension.
        x_k: Key tensor of shape [b, t, h, d].
        cos: Precomputed cosine values of shape [1, t, 1, d_half].
        sin: Precomputed sine values of shape [1, t, 1, d_half].

    Returns:
        Tuple of rotated (query, key) tensors, each of shape [b, t, h, d].
    """
    b, t, h, d = x_q.shape
    d_half = d // 2
    xq1, xq2 = x_q[..., :d_half], x_q[..., d_half:]
    xk1, xk2 = x_k[..., :d_half], x_k[..., d_half:]
    # rotate: (x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)
    xq_rot_1 = xq1 * cos - xq2 * sin
    xq_rot_2 = xq2 * cos + xq1 * sin
    xk_rot_1 = xk1 * cos - xk2 * sin
    xk_rot_2 = xk2 * cos + xk1 * sin
    x_q_out = torch.cat([xq_rot_1, xq_rot_2], dim=-1)
    x_k_out = torch.cat([xk_rot_1, xk_rot_2], dim=-1)
    return x_q_out, x_k_out

def build_rope_cache(t: int, d: int, device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds cosine and sine caches for Rotary Positional Embedding.

    Precomputes frequency-based rotational embeddings for efficient RoPE
    application during forward pass.

    Args:
        t: Sequence length (number of positions).
        d: Head dimension (must be even).
        device: Torch device to create tensors on.
        base: Base value for computing inverse frequencies. Larger values
            result in slower frequency decay.

    Returns:
        Tuple of (cos, sin) tensors, each of shape [1, t, 1, d/2] for
        correct broadcasting with [b, t, h, d/2].

    Raises:
        AssertionError: If d is not even.
    """
    assert d % 2 == 0, "head_dim must be even for RoPE"
    d_half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, d_half, device=device, dtype=torch.float32) / d_half))
    # positions 0..t-1
    positions = torch.arange(t, device=device, dtype=torch.float32)
    freqs = torch.einsum('t,f->tf', positions, inv_freq)  # [t, d_half]
    # Broadcast across batch and heads: [1, t, 1, d_half]
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)  # [1, t, 1, d_half]
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)  # [1, t, 1, d_half]
    return cos.to(dtype=torch.float32), sin.to(dtype=torch.float32)

# -----------------------------
# Gated Delta Rule with chunked scanning
# -----------------------------
class DeltaRuleGated(nn.Module):
    """Chunked recurrent delta memory with per-dimension forget and write gates.

    Implements a gated linear attention mechanism where memory state M ∈ R^{d×d}
    is updated recurrently with element-wise gating for both forgetting and writing.

    Update rule at time t for each head:
        M_t = (f_row_t ⊗ f_col_t) ⊙ M_{t-1} + (k_t ⊙ gk_t) (v_t ⊙ gv_t)^T
        o_t = q_t @ M_t

    Where f_row_t, f_col_t, gk_t, gv_t ∈ R^d with values in [f_min, 1] / [0, 1].

    Attributes:
        head_dim: Dimension of each attention head.
        chunk_size: Number of timesteps to process per chunk.
        f_min: Minimum forget gate value to prevent complete memory erasure.
    """

    def __init__(self, head_dim: int, chunk_size: int = 64, f_min: float = 0.8):
        """Initializes DeltaRuleGated module.

        Args:
            head_dim: Dimension of each attention head.
            chunk_size: Number of timesteps to process in each chunk.
            f_min: Minimum value for forget gate to ensure memory retention.
        """
        super().__init__()
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.f_min = f_min

    @torch.compile
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                f_gate: torch.Tensor,
                g_gate: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes gated delta rule attention with chunked scanning.

        Processes the input sequence in chunks, maintaining a per-head memory
        matrix that is updated with gated writes and reads via queries.

        Args:
            q: Query tensor of shape [b, t, h, d].
            k: Key tensor of shape [b, t, h, d].
            v: Value tensor of shape [b, t, h, d].
            f_gate: Forget gate tensor of shape [b, t, h, d] controlling
                memory retention per dimension.
            g_gate: Write gate tensor of shape [b, t, h, d] controlling
                write strength per dimension.
            attn_mask: Optional mask tensor of shape [b, t] where 1 indicates
                valid tokens and 0 indicates masked tokens.

        Returns:
            Output tensor of shape [b, t, h, d] containing the attention results.
        """
        b, t, h, d = q.shape
        assert d == self.head_dim

        # Prepare mask broadcast if provided
        if attn_mask is not None:
            # [b, t, 1, 1]
            am = attn_mask.to(q.dtype).unsqueeze(-1).unsqueeze(-1)
            q = q * am
            k = k * am
            v = v * am
            # For masked positions, avoid erasing memory: set forgetting to 1 and writing to 0
            f_gate = f_gate * am + (1.0 - am)  # where mask=0 -> f=1
            g_gate = g_gate * am               # where mask=0 -> g=0

        # Initialize outputs and state M per (b,h)
        out = torch.zeros_like(q)
        M = torch.zeros((b, h, d, d), device=q.device, dtype=q.dtype)

        # Process in chunks
        cs = self.chunk_size
        for s in range(0, t, cs):
            e = min(s + cs, t)
            q_ch = q[:, s:e]       # [b, l, h, d]
            k_ch = k[:, s:e]
            v_ch = v[:, s:e]
            f_ch = f_gate[:, s:e]
            g_ch = g_gate[:, s:e]
            l = e - s

            # Within-chunk sequential scan to respect recurrence
            for i in range(l):
                qi = q_ch[:, i]         # [b, h, d]
                ki = k_ch[:, i]
                vi = v_ch[:, i]
                fi = f_ch[:, i]         # [b, h, d]
                gi = g_ch[:, i]

                # Build per-dimension forget outer product
                # f_row ⊗ f_col -> [b,h,d,d]
                f_outer = fi.unsqueeze(-1) * fi.unsqueeze(-2)
                M = M * torch.clamp(f_outer, min=self.f_min, max=1.0)

                # Write update with gated k,v outer product
                k_g = ki * gi
                v_g = vi * gi
                # (b,h,d,1) @ (b,h,1,d) -> (b,h,d,d)
                delta = k_g.unsqueeze(-1) @ v_g.unsqueeze(-2)
                M = M + delta

                # Output: q @ M -> [b,h,d]
                oi = torch.einsum('bhd, bhde -> bhe', qi, M)
                out[:, s + i] = oi

        return out

# -----------------------------
# Local window attention (chunked)
# -----------------------------
class LocalCausalAttention(nn.Module):
    """Chunked local causal attention with fixed window size.

    Implements efficient local attention that only attends to a fixed window
    of previous tokens, providing O(N*W) complexity where W is window size.

    Attributes:
        head_dim: Dimension of each attention head.
        window_size: Maximum number of previous tokens to attend to.
        chunk_size: Number of timesteps to process per chunk.
        dropout: Dropout layer for attention weights.
    """

    def __init__(self, head_dim: int, window_size: int = 64, chunk_size: int = 64, dropout: float = 0.0):
        """Initializes LocalCausalAttention module.

        Args:
            head_dim: Dimension of each attention head.
            window_size: Maximum number of previous tokens to attend to.
            chunk_size: Number of timesteps to process in each chunk.
            dropout: Dropout probability for attention weights.
        """
        super().__init__()
        self.head_dim = head_dim
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.dropout = nn.Dropout(dropout)

    @torch.compile
    def forward(
                self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes chunked local causal attention limited to window_size tokens.

        For each query position, attends only to the previous window_size tokens
        (including itself), enabling efficient local attention with linear
        complexity in window size.

        Args:
            q: Query tensor of shape [b, t, h, d].
            k: Key tensor of shape [b, t, h, d].
            v: Value tensor of shape [b, t, h, d].
            attn_mask: Optional mask tensor of shape [b, t] where 1 indicates
                valid tokens and 0 indicates masked tokens.

        Returns:
            Output tensor of shape [b, t, h, d] containing local attention results.
        """
        b, t, h, d = q.shape
        W = self.window_size
        cs = self.chunk_size
        scale = 1.0 / math.sqrt(d)
        out = torch.zeros_like(q)

        # Prepare mask tensors if provided
        if attn_mask is not None:
            mask_bool = attn_mask.to(dtype=torch.bool)  # [b,t]
        else:
            mask_bool = None

        # Process in chunks along time
        for s in range(0, t, cs):
            e = min(s + cs, t)
            l = e - s
            # Iterate within chunk to respect causal windowing cleanly and avoid complex gather shapes
            for i in range(l):
                pos = s + i
                start_i = max(0, pos - (W - 1))
                end_i = pos + 1  # exclusive
                k_slice = k[:, start_i:end_i]   # [b, L, h, d]
                v_slice = v[:, start_i:end_i]   # [b, L, h, d]
                q_i = q[:, pos]                 # [b, h, d]

                # Compute scores: [b,h,L]
                scores = torch.einsum('bhd, blhd -> bhl', q_i, k_slice) * scale

                # Build key mask and query mask if provided
                if mask_bool is not None:
                    key_mask = mask_bool[:, start_i:end_i]                # [b,L]
                    key_mask = key_mask.unsqueeze(1).expand(-1, h, -1)    # [b,h,L]
                    q_valid = mask_bool[:, pos].unsqueeze(1).unsqueeze(2) # [b,1,1]
                else:
                    key_mask = None
                    q_valid = None

                # Masked softmax: numerically safe even when all keys are masked
                if key_mask is not None:
                    masked_scores = scores.masked_fill(~key_mask, -1e9)   # [b,h,L]
                    attn = torch.softmax(masked_scores, dim=-1)
                    attn = attn * key_mask.to(attn.dtype)                 # zero-out masked positions exactly
                else:
                    attn = torch.softmax(scores, dim=-1)

                if q_valid is not None:
                    # Zero out attention/output if the query itself is invalid (masked)
                    attn = attn * q_valid.to(attn.dtype)

                attn = self.dropout(attn)

                # Output: [b,h,d]
                out_i = torch.einsum('bhl, blhd -> bhd', attn, v_slice)

                # If query invalid, ensure output is exactly zero
                if q_valid is not None:
                    out_i = out_i * q_valid  # [b,1,1] broadcast over [b,h,d]

                out[:, pos] = out_i

        return out

# -----------------------------
# DeltaNet Layer with gating + RoPE + local attention fusion
# -----------------------------
class DeltaNetLayer(nn.Module):
    """DeltaNet attention layer with gating, RoPE, and local attention fusion.

    Combines gated linear delta memory with local causal attention to achieve
    both global context through the recurrent core and sharp local selectivity.

    Attributes:
        hidden_size: Total hidden dimension of the layer.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        q_proj: Query projection layer.
        k_proj: Key projection layer.
        v_proj: Value projection layer.
        f_proj: Forget gate projection layer.
        g_proj: Write gate projection layer.
        out_proj: Output projection layer.
        in_norm: Input RMS normalization.
        dropout: Dropout layer.
        layer_norm: Post-attention layer normalization.
        delta_core: Gated delta rule recurrent core.
        local_attn: Local causal attention module.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        """Initializes DeltaNetLayer.

        Args:
            hidden_size: Total hidden dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            **kwargs: Additional arguments including:
                chunk_size: Chunk size for both delta core and local attention.
                window_size: Window size for local attention.

        Raises:
            AssertionError: If hidden_size is not divisible by num_heads or
                if head_dim is not even (required for RoPE).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        assert self.head_dim % 2 == 0, "head_dim must be even to apply RoPE"

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Gates
        self.f_proj = nn.Linear(hidden_size, hidden_size, bias=True)  # forget gate per channel
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=True)  # write gate per channel
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Norms and dropout
        self.in_norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Components: recurrent core and local attention
        default_chunk = kwargs.get('chunk_size', 64)
        default_window = kwargs.get('window_size', 64)
        self.delta_core = DeltaRuleGated(self.head_dim, chunk_size=default_chunk, f_min=0.85)
        self.local_attn = LocalCausalAttention(self.head_dim, window_size=default_window, chunk_size=default_chunk, dropout=dropout)

        # init
        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes layer parameters with sensible defaults.

        Uses Xavier uniform initialization for projections and sets gate
        biases for high memory retention and conservative writing.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Gates biases for sensible defaults
        # Forget gate bias towards high retention (~0.97)
        p = 0.97
        bias_f = math.log(p / (1 - p))
        nn.init.constant_(self.f_proj.bias, bias_f)
        # Write gate bias slightly conservative
        nn.init.constant_(self.g_proj.bias, -0.5)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes forward pass of the DeltaNet attention layer.

        Applies gated delta memory and local attention in parallel, fuses
        their outputs, and returns the result with residual connection.

        Args:
            x: Input tensor of shape [b, t, hidden_size].
            mask: Optional attention mask of shape [b, t] where 1 indicates
                valid tokens and 0 indicates masked tokens.

        Returns:
            Output tensor of shape [b, t, hidden_size].
        """
        # Pre-norm for stability
        x_norm = self.in_norm(x)

        # Projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Gates (sigmoid bounds)
        f_gate = torch.sigmoid(self.f_proj(x_norm))  # [b, t, d_model]
        g_gate = torch.sigmoid(self.g_proj(x_norm))  # [b, t, d_model]

        # Reshape to heads
        # [b, t, h, d]
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)
        f_gate = rearrange(f_gate, 'b t (h d) -> b t h d', h=self.num_heads)
        g_gate = rearrange(g_gate, 'b t (h d) -> b t h d', h=self.num_heads)

        b, t, h, d = q.shape

        # RoPE
        cos, sin = build_rope_cache(t, d, device=x.device)
        # cast cos/sin to match dtype if needed
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # Recurrent delta core (head-parallel inside DeltaRuleGated)
        # Reorder to [b, t, h, d] expected by core
        delta_out = self.delta_core(q, k, v, f_gate, g_gate, attn_mask=mask)

        # Local attention branch for sharper selectivity
        local_out = self.local_attn(q, k, v, attn_mask=mask)

        # Fuse branches: simple sum (can be changed to gated fusion later)
        out = delta_out + local_out

        # Merge heads and project
        out = rearrange(out, 'b t h d -> b t (h d)')
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual and post-norm
        out = self.layer_norm(x + out)
        return out

# -----------------------------
# Transformer Block
# -----------------------------
class DeltaNetBlock(nn.Module):
    """Transformer block combining DeltaNet attention with feed-forward network.

    Attributes:
        attention: DeltaNetLayer for gated attention with local fusion.
        ffn: Feed-forward network with GELU activation.
        ffn_layer_norm: Layer normalization after FFN residual.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1, **kwargs):
        """Initializes DeltaNetBlock.

        Args:
            hidden_size: Hidden dimension of the block.
            num_heads: Number of attention heads.
            ffn_hidden_size: Hidden dimension of FFN. Defaults to 4 * hidden_size.
            dropout: Dropout probability.
            **kwargs: Additional arguments passed to DeltaNetLayer.
        """
        super().__init__()
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        self.attention = DeltaNetLayer(hidden_size, num_heads, dropout, **kwargs)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes forward pass through attention and FFN sub-layers.

        Args:
            x: Input tensor of shape [b, t, hidden_size].
            mask: Optional attention mask of shape [b, t].

        Returns:
            Output tensor of shape [b, t, hidden_size].
        """
        x = self.attention(x, mask)
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x

# -----------------------------
# DeltaNet Model (keeps class name compatibility via alias at bottom)
# -----------------------------
class DeltaNetModel(nn.Module):
    """Full DeltaNet language model with token and position embeddings.

    Implements a transformer-style architecture using DeltaNet blocks that
    combine gated linear delta memory with local causal attention.

    Attributes:
        hidden_size: Hidden dimension throughout the model.
        max_seq_len: Maximum supported sequence length.
        token_embedding: Token embedding layer.
        position_embedding: Learnable position embedding layer.
        dropout: Dropout layer for embeddings.
        layers: Stack of DeltaNetBlock layers.
        layer_norm: Final layer normalization.
        lm_head: Language modeling head (tied with token embeddings).
    """

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1, **kwargs):
        """Initializes DeltaNetModel.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Hidden dimension of the model.
            num_layers: Number of DeltaNetBlock layers.
            num_heads: Number of attention heads per layer.
            max_seq_len: Maximum sequence length for position embeddings.
            dropout: Dropout probability.
            **kwargs: Additional arguments passed to DeltaNetBlock.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads, dropout=dropout, **kwargs)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes embedding parameters with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes forward pass of the language model.

        Args:
            input_ids: Token indices of shape [b, t].
            attention_mask: Optional mask of shape [b, t] where 1 indicates
                valid tokens and 0 indicates padding.

        Returns:
            Logits tensor of shape [b, t, vocab_size].
        """
        b, t = input_ids.shape
        position_ids = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, -1)
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
        """Returns a human-readable summary of the model architecture.

        Returns:
            Multi-line string describing model configuration and parameters.
        """
        return f"""
DeltaNet Architecture Summary (Evolved):
- Core: Gated Linear Delta Memory + Local Causal Attention Fusion
- Hidden Size: {self.hidden_size}
- Layers: {len(self.layers)}
- Heads: {getattr(self.layers[0].attention, 'num_heads', 'N/A')}
- Max Sequence Length: {self.max_seq_len}
- Chunk Size (core/local): {getattr(self.layers[0].attention.delta_core, 'chunk_size', 'N/A')}/{getattr(self.layers[0].attention.local_attn, 'chunk_size', 'N/A')}
- Local Window: {getattr(self.layers[0].attention.local_attn, 'window_size', 'N/A')}
- Key Innovations: Per-channel gates, RoPE, chunked scan, O(NW) local attention
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""

# Alias to meet class name requirement
class DeltaNet(DeltaNetModel):
    """Alias for DeltaNetModel for backward compatibility."""

    pass

# Factory function remains compatible
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Factory function to create a DeltaNetModel with default configuration.

    Args:
        vocab_size: Size of the vocabulary.
        **kwargs: Override default configuration values. Supported keys:
            hidden_size (512), num_layers (6), num_heads (8),
            max_seq_len (2048), dropout (0.1).

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
    torch.set_float32_matmul_precision('high')
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    b, t = 2, 100
    input_ids = torch.randint(0, 1000, (b, t))
    attn_mask = torch.ones(b, t, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, attn_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
