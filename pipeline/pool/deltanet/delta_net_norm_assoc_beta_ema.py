"""
DeltaNet: Evolved Linear Attention Architecture with Normalized Associative Memory
- Implements per-head, input-dependent forgetting with stability clamps
- Adds EMA denominator normalization to prevent memory drift
- Chunkwise causal processing for efficiency and scalability
- Replaces view/reshape with einops.rearrange for robust dimension handling
- Maintains sub-quadratic complexity and batch-size agnosticism
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    Normalizes inputs per token along the last dimension using RMS.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """Initialize RMSNorm layer.

        Args:
            dim: The dimension of the input features to normalize.
            eps: Small constant for numerical stability. Defaults to 1e-6.
            elementwise_affine: If True, learns a per-element scaling weight.
                Defaults to True.
        """
        super().__init__()
        self.eps = eps
        self.dim = dim
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to input tensor.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            Normalized tensor of the same shape as input.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = x * rms
        if self.weight is not None:
            out = out * self.weight
        return out


class DeltaRule(nn.Module):
    """Baseline Delta rule for linear attention computation (legacy, unused in evolved DeltaNet)."""

    def __init__(self, hidden_size: int):
        """Initialize the DeltaRule module.

        Args:
            hidden_size: Dimension of the hidden state.
        """
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Compute delta rule attention with recurrent memory updates.

        Args:
            q: Query tensor of shape [batch_size, seq_len, hidden_size].
            k: Key tensor of shape [batch_size, seq_len, hidden_size].
            v: Value tensor of shape [batch_size, seq_len, hidden_size].
            beta: Decay factor tensor of shape [batch_size, seq_len].

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        batch_size, seq_len, hidden_size = q.shape
        h_t = torch.zeros(batch_size, hidden_size, hidden_size, device=q.device, dtype=q.dtype)
        outputs = []
        for t in range(seq_len):
            q_t = q[:, t:t + 1]
            k_t = k[:, t:t + 1]
            v_t = v[:, t:t + 1]
            beta_t = beta[:, t:t + 1]
            beta_expanded = beta_t.unsqueeze(-1)
            h_t = h_t * beta_expanded.squeeze(1)
            outer_product = torch.einsum('bik,bjk->bij', k_t, v_t)
            h_t = h_t + outer_product
            o_t = torch.einsum('bik,bkj->bij', q_t, h_t)
            outputs.append(o_t)
        return torch.cat(outputs, dim=1)


class DeltaNet(nn.Module):
    """Single DeltaNet attention layer with normalized associative memory and input-dependent forgetting.

    Forward signature preserved: forward(x: Tensor, mask: Optional[Tensor] = None) -> Tensor
    - x: [batch, seq_len, hidden_size]
    - mask: Optional [batch, seq_len] where 1 indicates valid, 0 indicates padding
    Returns: [batch, seq_len, hidden_size]
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, chunk_size: int = 128, **kwargs):
        """Initialize the DeltaNet attention layer.

        Args:
            hidden_size: Dimension of the model hidden state.
            num_heads: Number of attention heads. Defaults to 8.
            dropout: Dropout probability. Defaults to 0.1.
            chunk_size: Size of chunks for memory-efficient processing.
                Defaults to 128.
            **kwargs: Additional keyword arguments (unused).

        Raises:
            AssertionError: If hidden_size is not divisible by num_heads.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size

        # Pre-sublayer normalization (stabilizes Q/K/V projections)
        self.input_rmsnorm = RMSNorm(hidden_size)

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Post-projection RMSNorm for K/V to reduce covariate drift
        self.k_rmsnorm = RMSNorm(hidden_size)
        self.v_rmsnorm = RMSNorm(hidden_size)

        # Per-head, input-dependent forgetting gate
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # Per-head base decay parameter (initialized to long timescales with diversity)
        self.base_beta_logit = nn.Parameter(torch.zeros(num_heads))

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Residual and dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Stability clamps for forgetting dynamics
        self.register_buffer("beta_min", torch.tensor(0.90), persistent=False)
        self.register_buffer("beta_max", torch.tensor(0.9995), persistent=False)
        self.register_buffer("eps", torch.tensor(1e-6), persistent=False)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters.

        Uses Xavier uniform initialization for projection weights and
        configures beta projection for high retention (~0.99). Per-head
        base decays are initialized with multi-timescale diversity.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Initialize beta_proj bias towards high retention (~0.99)
        beta0 = 0.99
        beta0_logit = math.log(beta0 / (1.0 - beta0))
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, beta0_logit)

        # Initialize base per-head decays with multi-timescale diversity (0.96 .. 0.995)
        # Map heads to a log-uniform range using a simple schedule
        if self.base_beta_logit is not None:
            num = self.num_heads
            base_betas = []
            for h in range(num):
                # head 0 -> shorter memory, last head -> longest
                frac = (h + 0.5) / num
                beta_h = 0.96 * (1 - frac) + 0.995 * frac
                base_betas.append(beta_h)
            base_betas = torch.tensor(base_betas)
            base_logits = torch.log(base_betas / (1.0 - base_betas))
            with torch.no_grad():
                self.base_beta_logit.copy_(base_logits)

    @torch.compile
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform forward pass through the DeltaNet attention layer.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            mask: Optional attention mask of shape [batch, seq_len] where
                1 indicates valid tokens and 0 indicates padding.

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        b, s, d = x.shape

        # Residual connection
        residual = x

        # Pre-norm for stable projections
        x_norm = self.input_rmsnorm(x)

        # Project Q/K/V
        q = self.q_proj(x_norm)
        k = self.k_proj(self.k_rmsnorm(x_norm))
        v = self.v_proj(self.v_rmsnorm(x_norm))

        # Compute per-token, per-head forgetting (sigmoid in (0,1))
        beta_token = torch.sigmoid(self.beta_proj(x_norm))  # [b, s, h]

        # Apply base per-head decay
        base_beta = torch.sigmoid(self.base_beta_logit)  # [h]
        base_beta = base_beta.view(1, 1, self.num_heads)  # [1,1,h]
        beta = beta_token * base_beta  # [b,s,h]

        # Clamp to stability range using tensor min/max to avoid Python .item() inside compile
        beta_min = self.beta_min.to(dtype=beta.dtype, device=beta.device)
        beta_max = self.beta_max.to(dtype=beta.dtype, device=beta.device)
        beta = torch.clamp(beta, min=beta_min, max=beta_max)

        # Prepare optional attention mask: [b, s]
        m: Optional[torch.Tensor] = None
        if mask is not None:
            # Ensure float mask in {0,1}
            m = mask.to(dtype=beta.dtype).view(b, s)  # [b, s]
            m_exp = m.unsqueeze(-1)  # [b, s, 1]
            # For masked positions, prevent state update (set k/v to 0) and avoid decay (beta=1)
            beta = torch.where(m_exp > 0, beta, torch.ones_like(beta))
            k = k * m_exp
            v = v * m_exp

        # Reshape to multi-head: [b, s, h, d_h]
        q = rearrange(q, 'b s (h d_h) -> b s h d_h', h=self.num_heads)
        k = rearrange(k, 'b s (h d_h) -> b s h d_h', h=self.num_heads)
        v = rearrange(v, 'b s (h d_h) -> b s h d_h', h=self.num_heads)

        # Reorder to [b, h, s, d_h] for efficient per-head processing
        q = rearrange(q, 'b s h d_h -> b h s d_h')
        k = rearrange(k, 'b s h d_h -> b h s d_h')
        v = rearrange(v, 'b s h d_h -> b h s d_h')
        beta = rearrange(beta, 'b s h -> b h s')

        # Prepare head-expanded mask for denominator normalization if provided
        m_bhs: Optional[torch.Tensor] = None
        if m is not None:
            # [b, s] -> [b, h, s]
            m_bhs = m.unsqueeze(1).expand(-1, self.num_heads, -1)

        # Cast once to float32 for stability in the recurrent updates
        dtype_in = q.dtype
        q_f32 = q.to(dtype=torch.float32)
        k_f32 = k.to(dtype=torch.float32)
        v_f32 = v.to(dtype=torch.float32)

        # Initialize associative memory and denominator in float32 for stability
        M = torch.zeros((b, self.num_heads, self.head_dim, self.head_dim), device=x.device, dtype=torch.float32)
        denom = torch.zeros((b, self.num_heads, 1, 1), device=x.device, dtype=torch.float32)

        outputs = []  # will collect [b, h, chunk_len, d_h]

        one_f32 = torch.tensor(1.0, device=x.device, dtype=torch.float32)

        # Process in chunks to reduce memory without breaking causality
        for start in range(0, s, self.chunk_size):
            end = min(start + self.chunk_size, s)
            q_chunk_f = q_f32[:, :, start:end, :]  # [b,h,c,d_h]
            k_chunk_f = k_f32[:, :, start:end, :]
            v_chunk_f = v_f32[:, :, start:end, :]
            beta_chunk = beta[:, :, start:end]  # [b,h,c]

            # Mask chunk for denominator (valid-token indicator), if available
            if m_bhs is not None:
                m_chunk = m_bhs[:, :, start:end]  # [b,h,c]
            else:
                m_chunk = None

            c = end - start
            # Preallocate output chunk to avoid list appends inside compile
            out_chunk = torch.empty((b, self.num_heads, c, self.head_dim), device=x.device, dtype=dtype_in)

            for t in range(c):
                q_t_f = q_chunk_f[:, :, t, :]  # [b,h,d_h]
                k_t_f = k_chunk_f[:, :, t, :]
                v_t_f = v_chunk_f[:, :, t, :]
                beta_t = beta_chunk[:, :, t]  # [b,h]

                # Cast to float32 for updates
                beta_t_f = beta_t.to(dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)  # [b,h,1,1]

                # Denominator increment: add 1.0 only for valid tokens; 0.0 for padding steps
                if m_chunk is not None:
                    m_t = m_chunk[:, :, t]  # [b,h]
                    m_t_f = m_t.to(dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)  # [b,h,1,1]
                else:
                    m_t_f = one_f32  # scalar broadcast

                # Decay memory and denominator (EMA normalization)
                M = M * beta_t_f
                denom = denom * beta_t_f + m_t_f

                # Rank-1 outer product update: k_t^T v_t -> [b,h,d_h,d_h]
                kv_t = torch.einsum('bhd,bhe->bhde', k_t_f, v_t_f)
                M = M + kv_t

                # Retrieve with normalized memory
                M_norm = M / (denom + self.eps)
                o_t = torch.einsum('bhd,bhde->bhe', q_t_f, M_norm)  # [b,h,d_h]
                out_chunk[:, :, t, :] = o_t.to(dtype=dtype_in)

            outputs.append(out_chunk)

        # Concatenate chunks along sequence
        out = torch.cat(outputs, dim=2)  # [b,h,s,d_h]

        # Reassemble heads: [b, s, h, d_h] -> [b, s, d]
        out = rearrange(out, 'b h s d_h -> b s (h d_h)')

        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        return out


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block using evolved DeltaNet layer."""

    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        """Initialize a DeltaNet transformer block.

        Args:
            hidden_size: Dimension of the model hidden state.
            num_heads: Number of attention heads. Defaults to 8.
            ffn_hidden_size: Hidden size of the feed-forward network.
                Defaults to 4 * hidden_size if not specified.
            dropout: Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        self.attention = DeltaNet(hidden_size, num_heads, dropout)

        # Feed-forward network (kept simple; can be upgraded to SwiGLU later)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            mask: Optional attention mask of shape [batch, seq_len].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        x = self.attention(x, mask)
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings that adapt to any sequence length at runtime.
    Returns embeddings matching the provided positions tensor shape [b, s].
    """

    def __init__(self, dim: int, base: float = 10000.0):
        """Initialize sinusoidal positional embedding.

        Args:
            dim: Embedding dimension.
            base: Base for computing inverse frequencies. Defaults to 10000.0.
        """
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, positions: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Compute sinusoidal positional embeddings.

        Args:
            positions: Position indices tensor of shape [batch, seq_len].
            dtype: Optional output dtype. Defaults to float32.

        Returns:
            Positional embeddings of shape [batch, seq_len, dim].
        """
        b, s = positions.shape
        device = positions.device
        dim = self.dim
        # Compute inverse frequencies
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(self.base) / dim)
        )  # [dim//2]
        pos = positions.to(torch.float32)
        sinusoid_inp = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)  # [b, s, dim//2]
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        emb = torch.cat([sin, cos], dim=-1)  # [b, s, dim or dim-1]
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        if dtype is None:
            dtype = torch.float32
        emb = emb.to(dtype)
        return emb


class DeltaNetModel(nn.Module):
    """Complete DeltaNet model with token and position embeddings."""

    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6, num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
        """Initialize the DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Dimension of the model hidden state. Defaults to 512.
            num_layers: Number of transformer layers. Defaults to 6.
            num_heads: Number of attention heads. Defaults to 8.
            max_seq_len: Maximum sequence length for learned positional
                embeddings. Defaults to 2048.
            dropout: Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        # Add dynamic positional embedding fallback for sequences beyond max_seq_len
        self.sinusoidal_positions = SinusoidalPositionalEmbedding(hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([DeltaNetBlock(hidden_size, num_heads, dropout=dropout) for _ in range(num_layers)])

        # Output layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding parameters with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform forward pass through the model.

        Args:
            input_ids: Input token IDs of shape [batch, seq_len].
            attention_mask: Optional mask of shape [batch, seq_len] where
                1 indicates valid tokens and 0 indicates padding.

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size].
        """
        b, s = input_ids.shape
        pos_ids = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, -1)
        # Positional embeddings: use learned up to max_seq_len; otherwise, use sinusoidal to adapt
        if s <= self.max_seq_len:
            pos_emb = self.position_embedding(pos_ids)
        else:
            pos_emb = self.sinusoidal_positions(pos_ids, dtype=self.token_embedding.weight.dtype)
        # Token embeddings
        x = self.token_embedding(input_ids) + pos_emb
        x = self.dropout(x)
        # Apply layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        # Final norm and head
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        """Generate a summary of the model architecture.

        Returns:
            A formatted string describing the model configuration
            and parameter count.
        """
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Linear Attention Transformer with Normalized Associative Memory
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Input-dependent per-head decay, EMA normalization, chunkwise processing, RMSNorm
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model

def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Create a DeltaNet model with default or custom configuration.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257.
        **kwargs: Override default configuration values. Supported keys:
            hidden_size (int): Model hidden dimension. Defaults to 512.
            num_layers (int): Number of transformer layers. Defaults to 6.
            num_heads (int): Number of attention heads. Defaults to 8.
            max_seq_len (int): Maximum sequence length. Defaults to 2048.
            dropout (float): Dropout probability. Defaults to 0.1.

    Returns:
        An initialized DeltaNetModel instance.
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
    # Quick sanity test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=3, num_heads=8)
    b, s = 2, 3000  # test beyond max_seq_len to exercise dynamic positions
    input_ids = torch.randint(0, 1000, (b, s))
    attn_mask = torch.ones(b, s, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, attn_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Logits shape: {logits.shape}")
        print(model.get_architecture_summary())
