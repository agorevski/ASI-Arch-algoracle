"""
DeltaNet: Evolved Linear Attention Architecture with Multi-Timescale Low-Rank State and RoPE
This version introduces:
- Multi-timescale per-head per-rank retention (base decays) with content-dependent update gating
- Low-rank associative memory per head (rank << head_dim) for O(N * H * r * d_head)
- Chunkwise causal scanning for efficiency and memory locality
- Rotary Position Embeddings (RoPE) applied to Q/K for better length generalization
- Batch-size agnostic tensor ops using einops.rearrange (no .view/.reshape)
- Optional torch.compile on core chunk scanner for speed

It preserves:
- Model interfaces and forward signatures
- Sub-quadratic complexity
- Multi-head structure and residual/FFN block layout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange

# -----------------------------
# Utilities
# -----------------------------

def _logit(p: float) -> float:
    """Compute the logit (inverse sigmoid) of a probability.

    Args:
        p: Probability value to convert, will be clamped to (1e-6, 1-1e-6).

    Returns:
        The logit value log(p / (1 - p)).
    """
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _get_rope_cos_sin(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0):
    """Compute RoPE cos/sin matrices for last-dim rotations.

    Performs the internal trig computations in float32 for numerical stability
    and CPU compatibility, then casts to the requested dtype.

    Args:
        seq_len: Sequence length T.
        dim: Head dimension (head_dim).
        device: Torch device for the output tensors.
        dtype: Torch dtype for the output tensors.
        base: Base frequency for positional encoding. Defaults to 10000.0.

    Returns:
        A tuple (cos, sin, dim_rot) where:
            - cos: Cosine values with shape [seq_len, dim_rot // 2].
            - sin: Sine values with shape [seq_len, dim_rot // 2].
            - dim_rot: Number of dimensions that will be rotated (even number <= dim).
    """
    dim_rot = (dim // 2) * 2
    half_dim = dim_rot // 2
    if half_dim == 0:
        # No rotation for dim < 2
        cos = torch.ones(seq_len, 0, device=device, dtype=dtype)
        sin = torch.zeros(seq_len, 0, device=device, dtype=dtype)
        return cos, sin, dim_rot
    # Use float32 for cos/sin to avoid half-precision CPU trig issues and improve stability
    comp_dtype = torch.float32
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, device=device, dtype=comp_dtype) / half_dim))
    t = torch.arange(seq_len, device=device, dtype=comp_dtype)
    freqs = torch.einsum('t,f->tf', t, inv_freq)  # [T, half_dim]
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin, dim_rot


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dim_rot: int) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to the last dimension of x.

    Only the first dim_rot dimensions (even-indexed pairs) are rotated;
    remaining dimensions are left unchanged.

    Args:
        x: Input tensor with shape [B, T, H, D].
        cos: Cosine values with shape [T, half_dim].
        sin: Sine values with shape [T, half_dim].
        dim_rot: Number of dimensions to rotate (must be even).

    Returns:
        Tensor with same shape as x, with RoPE applied to the first dim_rot dimensions.
    """
    B, T, H, D = x.shape
    if dim_rot == 0:
        return x
    half_dim = dim_rot // 2
    # Ensure cos/sin are in the same dtype as x for safe mixed-precision behavior
    if cos.dtype != x.dtype:
        cos = cos.to(x.dtype)
    if sin.dtype != x.dtype:
        sin = sin.to(x.dtype)
    # Prepare cos/sin broadcast: [1, T, 1, half_dim]
    cos_b = rearrange(cos, 't f -> 1 t 1 f')
    sin_b = rearrange(sin, 't f -> 1 t 1 f')

    x_rot = x[..., :dim_rot]
    x_pass = x[..., dim_rot:]
    x1 = x_rot[..., 0:dim_rot:2]
    x2 = x_rot[..., 1:dim_rot:2]
    # Rotate
    x_rotated_1 = x1 * cos_b - x2 * sin_b
    x_rotated_2 = x1 * sin_b + x2 * cos_b
    x_rotated = torch.stack((x_rotated_1, x_rotated_2), dim=-1)
    x_rotated = rearrange(x_rotated, 'b t h f two -> b t h (f two)')
    return torch.cat([x_rotated, x_pass], dim=-1)


# -----------------------------
# Evolved DeltaNet Layer
# -----------------------------

class DeltaNet(nn.Module):
    """
    DeltaNet Layer: Multi-timescale low-rank linear attention with RoPE and chunked causal scan.
    Preserves forward signature: forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        rank: Optional[int] = None,
        chunk_size: int = 64,
        rope_base: float = 10000.0,
        use_rope: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        self.head_dim = hidden_size // num_heads
        # Low-rank for per-head associative memory
        if rank is None:
            # Sensible default: 1/4 of head_dim (at least 4)
            rank = max(4, self.head_dim // 4)
        self.rank = rank
        self.chunk_size = chunk_size
        self.rope_base = rope_base
        self.use_rope = use_rope

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Low-rank per-head projection parameters: W_q, W_k in R^{H, r, d_head}
        self.Wq = nn.Parameter(torch.empty(self.num_heads, self.rank, self.head_dim))
        self.Wk = nn.Parameter(torch.empty(self.num_heads, self.rank, self.head_dim))

        # Content-dependent update gate per head-rank
        self.g_proj = nn.Linear(hidden_size, self.num_heads * self.rank, bias=True)

        # Per-head-per-rank base retention (decay in (0,1)) as logits
        self.base_decay_logits = nn.Parameter(torch.empty(self.num_heads, self.rank))

        # Output projection and regularization
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self._reset_parameters()

        # Compile chunk scanner lazily; define as attribute to allow torch.compile
        self._compiled = False

    def _reset_parameters(self):
        """Initialize all learnable parameters.

        Uses Xavier uniform initialization for projection weights and sets
        base decay logits for high retention (~0.98). Gate biases are
        initialized to prefer small updates initially.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)

        # Initialize decays around high retention (e.g., ~0.98)
        with torch.no_grad():
            self.base_decay_logits.copy_(torch.full_like(self.base_decay_logits, _logit(0.98)))
        # Initialize gate bias to prefer small updates initially
        nn.init.constant_(self.g_proj.bias, _logit(0.1))

    def _chunk_scan_impl(
        self,
        q_r: torch.Tensor,
        k_r: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        base_decay: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        M_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform chunked causal scan with low-rank associative memory.

        Iterates through each timestep in the chunk, updating the memory state
        with gated outer products and computing outputs via query-memory readout.

        Args:
            q_r: Low-rank query projections with shape [B, T_c, H, r].
            k_r: Low-rank key projections with shape [B, T_c, H, r].
            v: Value tensor with shape [B, T_c, H, D].
            g: Update gates in [0, 1] with shape [B, T_c, H, r].
            base_decay: Base retention decay in (0, 1) with shape [H, r].
            mask: Optional mask with shape [B, T_c], 1 for valid, 0 for padding.
            M_init: Optional initial memory state with shape [B, H, r, D].

        Returns:
            A tuple (Y, M) where:
                - Y: Output tensor with shape [B, T_c, H, D].
                - M: Final memory state with shape [B, H, r, D].
        """
        B, T_c, H, r = q_r.shape
        D = v.shape[-1]
        # State per batch, head: [B, H, r, D]
        if M_init is None:
            M = v.new_zeros(B, H, r, D)
        else:
            M = M_init
        outputs = []
        # Precompute broadcasted decay
        decay = base_decay.unsqueeze(0).unsqueeze(-1)  # [1, H, r, 1]
        scale = 1.0 / math.sqrt(D)  # update scaling for stability (use D to balance v magnitude)
        for t in range(T_c):
            q_t = q_r[:, t]      # [B, H, r]
            k_t = k_r[:, t]      # [B, H, r]
            v_t = v[:, t]        # [B, H, D]
            g_t = g[:, t]        # [B, H, r]
            # Optional mask at t
            if mask is not None:
                # float mask in {0,1}
                m_t_scalar = mask[:, t].to(dtype=v.dtype)
                m_t4 = m_t_scalar.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,1,1,1] for state update
                m_t3 = m_t_scalar.unsqueeze(-1).unsqueeze(-1)                # [B,1,1] for output
            # Update: outer product (B,H,r) x (B,H,D) -> (B,H,r,D)
            upd = torch.einsum('bhr,bhd->bhrd', (g_t * k_t) * scale, v_t)
            if mask is not None:
                # New state if token is valid; otherwise keep previous state (no decay, no update)
                new_M = M * decay + upd
                M = torch.where(m_t4.bool(), new_M, M)
            else:
                # Unmasked: always decay and update
                M = M * decay + upd
            # Readout: (B,H,r) x (B,H,r,D) -> (B,H,D)
            y_t = torch.einsum('bhr,bhrd->bhd', q_t, M)
            if mask is not None:
                y_t = y_t * m_t3
            outputs.append(y_t)
        Y = torch.stack(outputs, dim=1)  # [B, T_c, H, D]
        return Y, M

    def _maybe_compile(self):
        """Lazily compile the chunk scan implementation with torch.compile.

        Attempts to compile _chunk_scan_impl for improved performance.
        Fails silently if torch.compile is not available.
        """
        if not self._compiled:
            try:
                self._chunk_scan_impl = torch.compile(self._chunk_scan_impl, mode='default', fullgraph=False)
            except Exception:
                # Fallback silently if torch.compile not available
                pass
            self._compiled = True

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the DeltaNet attention layer.

        Applies multi-head low-rank linear attention with RoPE, content-gated
        updates, multi-timescale retention, and chunked causal scanning.

        Args:
            x: Input tensor with shape [batch, seq_len, hidden_size].
            mask: Optional mask with shape [batch, seq_len], where 1 indicates
                valid tokens and 0 indicates padding.

        Returns:
            Output tensor with shape [batch, seq_len, hidden_size].
        """
        B, T, Hs = x.shape
        assert Hs == self.hidden_size
        residual = x

        # Normalize/standardize mask shape to [B, T] if provided as [T, B]
        if mask is not None and mask.dim() == 2:
            if mask.shape[0] == T and mask.shape[1] == B:
                mask = mask.t()

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)

        # Apply RoPE on q and k
        if self.use_rope:
            cos, sin, dim_rot = _get_rope_cos_sin(T, self.head_dim, device=x.device, dtype=x.dtype, base=self.rope_base)
            q = _apply_rope(q, cos, sin, dim_rot)
            k = _apply_rope(k, cos, sin, dim_rot)

        # Low-rank projections per head: q_r, k_r in [B, T, H, r]
        # q_r(b,t,h,r) = sum_d q(b,t,h,d) * Wq(h,r,d)
        q_r = torch.einsum('bthd,hrd->bthr', q, self.Wq)
        k_r = torch.einsum('bthd,hrd->bthr', k, self.Wk)

        # Content-dependent update gates in [0,1], shape [B, T, H, r]
        g = torch.sigmoid(rearrange(self.g_proj(x), 'b t (h r) -> b t h r', h=self.num_heads))
        if mask is not None:
            # Ensure float mask in {0,1}
            m = mask.to(dtype=x.dtype)
            g = g * rearrange(m, 'b t -> b t 1 1')

        # Base decay per head-rank
        base_decay = torch.sigmoid(self.base_decay_logits)  # [H, r]

        # Chunked causal scan with state carried across chunks
        self._maybe_compile()
        outputs = []
        M_state: Optional[torch.Tensor] = None
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            q_rs = q_r[:, start:end]
            k_rs = k_r[:, start:end]
            v_s = v[:, start:end]
            g_s = g[:, start:end]
            mask_s = mask[:, start:end] if mask is not None else None
            y_s, M_state = self._chunk_scan_impl(q_rs, k_rs, v_s, g_s, base_decay, mask_s, M_state)
            outputs.append(y_s)
        y = torch.cat(outputs, dim=1)  # [B, T, H, D]

        # Merge heads and project out
        y = rearrange(y, 'b t h d -> b t (h d)')
        y = self.out_proj(y)
        y = self.dropout(y)
        y = self.layer_norm(residual + y)
        return y


# -----------------------------
# Transformer Block and Model
# -----------------------------

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
        """Forward pass through the complete DeltaNet transformer block.

        Applies self-attention followed by a feed-forward network with
        residual connections and layer normalization.

        Args:
            x: Input tensor with shape [batch, seq_len, hidden_size].
            mask: Optional attention mask with shape [batch, seq_len].

        Returns:
            Output tensor with shape [batch, seq_len, hidden_size].
        """
        # Self-attention
        x = self.attention(x, mask)

        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)

        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet model"""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
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

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the complete DeltaNet model.

        Computes token and position embeddings, applies transformer layers,
        and projects to vocabulary logits.

        Args:
            input_ids: Token IDs with shape [batch, seq_len].
            attention_mask: Optional attention mask with shape [batch, seq_len].

        Returns:
            Logits tensor with shape [batch, seq_len, vocab_size].
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
            A formatted string describing the model's configuration,
            including hidden size, layers, heads, rank, and parameter count.
        """
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Low-Rank Linear Attention Transformer with RoPE
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Head Dim: {self.layers[0].attention.head_dim}
- Rank per Head: {self.layers[0].attention.rank}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Multi-timescale retention, content-gated low-rank memory, chunked causal scan, RoPE
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Create a DeltaNet model with default parameters.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257.
        **kwargs: Additional model parameters to override defaults.
            Supported keys: hidden_size, num_layers, num_heads,
            max_seq_len, dropout.

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
    # Smoke test
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=2, num_heads=4)
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
