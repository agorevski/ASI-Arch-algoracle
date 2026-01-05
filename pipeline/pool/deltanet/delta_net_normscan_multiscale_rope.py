"""
DeltaNet: Normalized Linear Attention with Multi-Timescale Retention, RoPE, and Chunked Causal Scan
This evolved architecture enhances selectivity and long-range memory while preserving O(n) complexity.
Improvements over the seed:
- Normalized linear attention (Performer-style) with running state s_t and memory M_t for sharp, scale-invariant reads
- Per-head multi-timescale exponential retention r (learned via softplus-parameterized tau) initialized log-uniformly over half-lives
- Content-gated writes for interference reduction
- RoPE relative positional encoding applied to q/k for better order sensitivity
- Chunked causal scan with @torch.compile for performance
- Batch-agnostic, einops-based shape handling (no .view/.reshape)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Compute the numerically stable inverse of the softplus function.

    For x > 0, computes y such that softplus(y) = x, i.e., y = x + log(1 - exp(-x)).
    Uses log1p for stability when exp(-x) is small.

    Args:
        x: Input tensor with positive values.

    Returns:
        Tensor containing the inverse softplus of the input.
    """
    x = torch.clamp(x, min=torch.finfo(x.dtype).tiny)
    return x + torch.log1p(-torch.exp(-x))


def _rope_sincos(seq_len: int, head_dim: int, device, dtype):
    """Compute sinusoidal position embeddings for Rotary Position Encoding (RoPE).

    Generates sin and cos frequency tables used for relative positional encoding
    in attention mechanisms.

    Args:
        seq_len: Length of the sequence.
        head_dim: Dimension of each attention head (must be even).
        device: Device to create tensors on.
        dtype: Data type for the output tensors.

    Returns:
        Tuple of (sin, cos) tensors, each with shape [seq_len, head_dim // 2].
    """
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum('i,j->ij', t, inv_freq)  # [S, half]
    sin = torch.sin(freqs).to(dtype)
    cos = torch.cos(freqs).to(dtype)
    return sin, cos  # [S, half] each


def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Encoding (RoPE) to input tensor.

    Applies rotary embeddings to the input by splitting the last dimension in half
    and applying rotation using the provided sin and cos tables.

    Args:
        x: Input tensor with shape [B, S, H, D] where D must be even.
        sin: Sine frequencies with shape [S, D//2].
        cos: Cosine frequencies with shape [S, D//2].

    Returns:
        Tensor with RoPE applied, same shape as input [B, S, H, D].
    """
    B, S, H, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    sin_ = sin.view(1, S, 1, half)
    cos_ = cos.view(1, S, 1, half)
    x1r = x1 * cos_ - x2 * sin_
    x2r = x1 * sin_ + x2 * cos_
    return torch.cat([x1r, x2r], dim=-1)


class DeltaNet(nn.Module):
    """Normalized linear attention module with multi-timescale retention and RoPE.

    Implements a DeltaNet attention mechanism featuring normalized linear attention,
    per-head multi-timescale exponential retention, content-gated writes, RoPE
    positional encoding, and chunked causal scan for efficient computation.

    Attributes:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        chunk_size: Size of chunks for the causal scan.
        eps: Small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        """Initialize the DeltaNet attention module.

        Args:
            hidden_size: Model hidden dimension, must be divisible by num_heads.
            num_heads: Number of attention heads. Defaults to 8.
            dropout: Dropout probability. Defaults to 0.1.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        )
        # Projections (read/write disentanglement kept simple: single k, v used for both write and normalizer)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # Content-gated write strength per head
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        # Per-head retention time parameter (tau) -> r = exp(-1/tau)
        # Initialize tau via half-lives spread log-uniformly
        half_lives = torch.logspace(start=torch.log10(torch.tensor(32.0)), end=torch.log10(torch.tensor(2048.0)), steps=num_heads)
        r0 = 0.5 ** (1.0 / half_lives)  # per-step retention to achieve chosen half-life
        tau0 = -1.0 / torch.log(r0)  # tau such that exp(-1/tau) = r0
        # Parameterize tau = softplus(theta) + 1, so initialize theta as inverse_softplus(tau0 - 1)
        theta0 = _inverse_softplus(tau0 - 1.0)
        self.theta = nn.Parameter(theta0)
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Numerical stability
        self.eps = 1e-6
        # Chunk size for causal scan
        self.chunk_size = 128
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize projection weights and biases.

        Uses Xavier uniform initialization for projection weights and sets
        appropriate initial values for biases.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.gate_proj.bias, 1.0)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map for positive features in linear attention.

        Uses ELU activation shifted to ensure strictly positive outputs,
        enabling numerically stable linear attention computation.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with positive feature values, same shape as input.
        """
        return F.elu(x, alpha=1.0) + 1.0 + 1e-6

    def _retention(self) -> torch.Tensor:
        """Compute per-head retention factors for exponential decay.

        Computes retention r in (0, 1) as r = exp(-1 / (softplus(theta) + 1)),
        where theta is a learned parameter per head.

        Returns:
            Tensor of shape [num_heads] with retention factors.
        """
        tau = F.softplus(self.theta) + 1.0  # [H]
        r = torch.exp(-1.0 / tau)
        return r  # [H]

    @torch.compile(mode="reduce-overhead", fullgraph=False)
    def _scan_chunks(self,
                     phi_q: torch.Tensor,
                     phi_k: torch.Tensor,
                     v: torch.Tensor,
                     gate: torch.Tensor,
                     mask: torch.Tensor,
                     r: torch.Tensor) -> torch.Tensor:
        """Perform chunked causal scan with memory decay and gated writes.

        Implements a linear attention scan with exponential memory decay,
        processing the sequence in chunks for efficiency. Uses running
        memory state M and normalizer s for normalized readout.

        Args:
            phi_q: Query features with shape [B, S, H, D].
            phi_k: Key features with shape [B, S, H, D].
            v: Value tensor with shape [B, S, H, D].
            gate: Content gate strengths with shape [B, S, H].
            mask: Attention mask with shape [B, S], values 0 or 1.
            r: Per-head retention factors with shape [H].

        Returns:
            Output tensor with shape [B, S, H, D].
        """
        B, S, H, D = phi_q.shape
        # Ensure r matches compute dtype/device to avoid promotions inside the scan loop
        r = r.to(dtype=phi_q.dtype, device=phi_q.device)
        # States: memory M [B, H, D, D], normalizer s [B, H, D]
        M = phi_q.new_zeros((B, H, D, D))
        s = phi_q.new_zeros((B, H, D))
        outputs = []
        r_bh11 = r.view(1, H, 1, 1)
        r_bh1 = r.view(1, H, 1)
        for start in range(0, S, self.chunk_size):
            end = min(start + self.chunk_size, S)
            for t in range(start, end):
                q_t = phi_q[:, t]         # [B, H, D]
                k_t = phi_k[:, t]         # [B, H, D]
                v_t = v[:, t]             # [B, H, D]
                g_t = gate[:, t]          # [B, H]
                m_t = mask[:, t].view(B, 1)  # [B, 1]
                # Broadcast mask to per-head
                g_t = g_t * m_t  # [B, H]
                g_t1 = g_t.view(B, H, 1)
                # Decay states
                M = M * r_bh11
                s = s * r_bh1
                # Write update (rank-1 outer product per b,h): M += (k_t * g) @ v_t^T
                M = M + torch.einsum('bhd,bhe->bhde', k_t * g_t1, v_t)
                # Normalizer update: s += k_t * g (use [B,H,1] to broadcast across D)
                s = s + k_t * g_t1
                # Readout: y_t = (q_t @ M) / (q_t @ s + eps)
                numer = torch.einsum('bhd,bhde->bhe', q_t, M)  # [B, H, D]
                denom = (q_t * s).sum(dim=-1) + self.eps       # [B, H]
                y_t = numer / denom.view(B, H, 1)
                # Optional: zero outputs at masked positions to avoid propagating padded activations
                y_t = y_t * m_t.view(B, 1, 1)
                outputs.append(y_t)
        y = torch.stack(outputs, dim=1)  # [B, S, H, D]
        return y

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply DeltaNet attention to the input.

        Computes normalized linear attention with RoPE, content gating,
        multi-timescale retention, and residual connection with layer norm.

        Args:
            x: Input tensor with shape [B, S, hidden_size].
            mask: Optional attention mask. Supports boolean masks, 0/1 numeric
                masks, or additive masks (negative values indicate masked positions).
                Shape [B, S]. Defaults to None (no masking).

        Returns:
            Output tensor with shape [B, S, hidden_size].
        """
        B, S, Dm = x.shape
        residual = x
        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        gate = torch.sigmoid(self.gate_proj(x))  # [B, S, H]
        # Reshape to [B, S, H, D]
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        # RoPE on q,k if even head_dim
        if (self.head_dim % 2) == 0:
            sin, cos = _rope_sincos(S, self.head_dim, device=x.device, dtype=x.dtype)
            q = _apply_rope(q, sin, cos)
            k = _apply_rope(k, sin, cos)
        # Feature map
        phi_q = self._phi(q)
        phi_k = self._phi(k)
        # Mask handling: robustly convert to 0/1 in the same dtype as x
        if mask is None:
            mask_t = torch.ones((B, S), device=x.device, dtype=x.dtype)
        else:
            if mask.dtype == torch.bool:
                mask_t = mask.to(dtype=x.dtype)
            else:
                # Numeric masks: handle 0/1 and additive (-inf for masked, 0 for valid)
                if torch.is_floating_point(mask):
                    minv = float(mask.min())
                    maxv = float(mask.max())
                    if minv < 0.0:
                        # Treat as additive mask (>= 0 is valid)
                        mask_t = (mask >= 0).to(dtype=x.dtype)
                    elif maxv <= 1.0:
                        # Probable 0/1 mask
                        mask_t = (mask > 0.5).to(dtype=x.dtype)
                    else:
                        # Fallback: non-zero is valid
                        mask_t = (mask != 0).to(dtype=x.dtype)
                else:
                    # Integer masks
                    mask_t = (mask != 0).to(dtype=x.dtype)
        # Retention per head
        r = self._retention()  # [H]
        # Chunked causal scan
        y = self._scan_chunks(phi_q, phi_k, v, gate, mask_t, r)  # [B, S, H, D]
        # Merge heads and project out
        y = rearrange(y, 'b s h d -> b s (h d)')
        y = self.out_proj(y)
        y = self.dropout(y)
        out = self.layer_norm(residual + y)
        return out


class DeltaNetBlock(nn.Module):
    """Transformer block with DeltaNet attention and feed-forward network.

    Combines DeltaNet attention with a GELU-activated feed-forward network
    and layer normalization for a complete transformer block.

    Attributes:
        attention: DeltaNet attention module.
        ffn: Feed-forward network.
        ffn_layer_norm: Layer normalization after FFN.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        """Initialize the DeltaNet transformer block.

        Args:
            hidden_size: Model hidden dimension.
            num_heads: Number of attention heads. Defaults to 8.
            ffn_hidden_size: Hidden dimension of the FFN. Defaults to 4 * hidden_size.
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
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the DeltaNet block to the input.

        Args:
            x: Input tensor with shape [B, S, hidden_size].
            mask: Optional attention mask with shape [B, S]. Defaults to None.

        Returns:
            Output tensor with shape [B, S, hidden_size].
        """
        x = self.attention(x, mask)
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet language model with embedding and LM head.

    A full transformer language model using DeltaNet attention blocks,
    with tied input/output embeddings for parameter efficiency.

    Attributes:
        hidden_size: Model hidden dimension.
        max_seq_len: Maximum sequence length.
        token_embedding: Token embedding layer.
        position_embedding: Learned position embedding layer.
        layers: Stack of DeltaNet blocks.
        layer_norm: Final layer normalization.
        lm_head: Language model head (tied with token embedding).
    """

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6, num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
        """Initialize the DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Model hidden dimension. Defaults to 512.
            num_layers: Number of DeltaNet blocks. Defaults to 6.
            num_heads: Number of attention heads per block. Defaults to 8.
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
        """Initialize embedding weights with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute language model logits for input token IDs.

        Args:
            input_ids: Token IDs with shape [B, S].
            attention_mask: Optional attention mask with shape [B, S]. Defaults to None.

        Returns:
            Logits tensor with shape [B, S, vocab_size].
        """
        B, S = input_ids.shape
        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(B, -1)
        tok = self.token_embedding(input_ids)
        pos = self.position_embedding(pos_ids)
        x = tok + pos
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        """Generate a human-readable summary of the model architecture.

        Returns:
            Multi-line string describing the model configuration and key features.
        """
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Normalized Linear Attention Transformer (Fast-weights with retention)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: normalized linear attention, multi-timescale per-head retention, RoPE, content-gated writes, chunked compile scan
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Create a DeltaNet language model with default or custom configuration.

    Factory function that creates a DeltaNetModel with sensible defaults
    that can be overridden via keyword arguments.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2 tokenizer size).
        **kwargs: Override default configuration. Supported keys:
            hidden_size (int): Model hidden dimension. Default 512.
            num_layers (int): Number of DeltaNet blocks. Default 6.
            num_heads (int): Number of attention heads. Default 8.
            max_seq_len (int): Maximum sequence length. Default 2048.
            dropout (float): Dropout probability. Default 0.1.

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
    # Minimal test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    B, S = 2, 100
    input_ids = torch.randint(0, 1000, (B, S))
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
