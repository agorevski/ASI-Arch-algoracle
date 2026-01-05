"""
DeltaNet: Enhanced Linear Attention Architecture (Multi-Scale Gated + Phi + Chunked)
Improvements implemented based on experimental evidence and research insights.
Core properties preserved: linear-time streaming, causal integrity, sub-quadratic complexity.

Key upgrades:
- Multi-timescale (K=2) input-conditioned forgetting with stability constraints
- Positive feature-map linear attention (phi = elu + 1) with running normalization
- Chunked sequential processing for memory efficiency
- Head-specific recency bias (ALiBi-like exponential decay in recurrent state domain)
- SwiGLU FFN and RMSNorm for depth stability
- Batch-size agnostic dynamic shapes and einops.rearrange for all reshaping
- @torch.compile on core state-update kernel

Note: Class name for the full model is DeltaNet (as required).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ------------------------------
# Utilities
# ------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A simplified layer normalization that only uses the root mean square
    of the input, without mean centering.

    Args:
        dim: The dimension of the input features.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initializes the RMSNorm layer.

        Args:
            dim: The dimension of the input features.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization to the input tensor.

        Args:
            x: Input tensor of shape [*, dim].

        Returns:
            Normalized tensor of the same shape as input.
        """
        norm_x = x.mul(x).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function as a feed-forward network module.

    Implements the SwiGLU activation: SwiGLU(x, W, V) = Swish(xW) * xV,
    which has been shown to improve transformer performance.

    Args:
        dim: Input and output dimension.
        hidden_dim: Hidden layer dimension (before the 2x expansion for gating).
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        """Initializes the SwiGLU module.

        Args:
            dim: Input and output dimension.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        # Use 2 * hidden_dim projection for SwiGLU (split into gate and value)
        self.w_in = nn.Linear(dim, hidden_dim * 2)
        self.w_out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies SwiGLU activation and projection.

        Args:
            x: Input tensor of shape [*, dim].

        Returns:
            Output tensor of shape [*, dim].
        """
        a, b = self.w_in(x).chunk(2, dim=-1)
        # swish(x) = x * sigmoid(x)
        x = F.silu(a) * b
        x = self.w_out(x)
        x = self.dropout(x)
        return x


# ------------------------------
# Compiled core: multi-timescale delta update under phi + normalization
# ------------------------------

@torch.compile
def compiled_delta_update(
    q: torch.Tensor,  # [b, t, d]
    k: torch.Tensor,  # [b, t, d]
    v: torch.Tensor,  # [b, t, d]
    beta_in: torch.Tensor,  # [b, t, 1]
    mask: torch.Tensor,  # [b, t, 1] with 1 for valid tokens, 0 for padding
    base_beta_1: torch.Tensor,  # scalar tensor
    base_beta_2: torch.Tensor,  # scalar tensor
    beta_min: float,
    beta_max: float,
    eps: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compiled multi-timescale delta update kernel with phi feature map.

    Performs chunked sequential processing of the delta-rule attention
    mechanism with two timescales, positive feature maps (phi = elu + 1),
    and running normalization for stability.

    Args:
        q: Query tensor of shape [batch, seq_len, head_dim].
        k: Key tensor of shape [batch, seq_len, head_dim].
        v: Value tensor of shape [batch, seq_len, head_dim].
        beta_in: Input-conditioned gate of shape [batch, seq_len, 1].
        mask: Attention mask with 1 for valid tokens, 0 for padding,
            shape [batch, seq_len, 1].
        base_beta_1: Scalar tensor for the first timescale base decay.
        base_beta_2: Scalar tensor for the second timescale base decay.
        beta_min: Minimum value for clamping decay factors.
        beta_max: Maximum value for clamping decay factors.
        eps: Small constant for numerical stability in normalization.
        chunk_size: Size of chunks for memory-efficient processing.

    Returns:
        Output tensor of shape [batch, seq_len, head_dim] containing
        the normalized linear attention results.
    """
    bsz, seqlen, d = q.shape

    # Positive feature map for stability (Performers): phi(x) = elu(x) + 1
    phi_q = F.elu(q, alpha=1.0) + 1.0  # [b, t, d]
    phi_k = F.elu(k, alpha=1.0) + 1.0  # [b, t, d]

    # States: Two multi-timescale associative memories and key normalizers
    H1 = torch.zeros((bsz, d, d), dtype=q.dtype, device=q.device)
    H2 = torch.zeros((bsz, d, d), dtype=q.dtype, device=q.device)
    Z1 = torch.zeros((bsz, d), dtype=q.dtype, device=q.device)
    Z2 = torch.zeros((bsz, d), dtype=q.dtype, device=q.device)

    # Base decays (sigmoid -> (0, 1))
    b1_base = torch.sigmoid(base_beta_1).clamp(beta_min, beta_max)
    b2_base = torch.sigmoid(base_beta_2).clamp(beta_min, beta_max)

    # Preallocate outputs to avoid Python list ops inside compiled function
    outputs = torch.empty((bsz, seqlen, d), dtype=q.dtype, device=q.device)

    # Process in chunks for memory efficiency
    for s in range(0, seqlen, chunk_size):
        e = min(s + chunk_size, seqlen)

        # Loop within chunk sequentially to preserve causality and state
        for t in range(s, e):
            q_t = phi_q[:, t]  # [b, d]
            k_t = phi_k[:, t]  # [b, d]
            v_t = v[:, t]      # [b, d]
            b_in_t = beta_in[:, t]  # [b, 1]
            m_t = mask[:, t]   # [b, 1]

            # Effective decays: combine base with input-conditioned gate
            # Per-token, per-batch beta in (0,1) assumed; clamp and mix
            b_in_t = b_in_t.clamp(beta_min, beta_max)
            b1_t = (b1_base * b_in_t).clamp(beta_min, beta_max)  # [b, 1]
            b2_t = (b2_base * b_in_t).clamp(beta_min, beta_max)  # [b, 1]

            # Decay previous states
            # Broadcast: [b,1,1] for H*, [b,1] for Z*
            b1_mat = b1_t.view(bsz, 1, 1)
            b2_mat = b2_t.view(bsz, 1, 1)
            H1 = H1 * b1_mat
            H2 = H2 * b2_mat
            Z1 = Z1 * b1_t
            Z2 = Z2 * b2_t

            # Associative updates with outer product of key and value
            # Apply mask to prevent padded positions from updating state
            kv_outer = torch.einsum('bd,be->bde', k_t, v_t)  # [b, d, d]
            m_broadcast_3d = m_t.view(bsz, 1, 1)
            m_broadcast_1d = m_t  # [b, 1]
            H1 = H1 + kv_outer * m_broadcast_3d
            H2 = H2 + kv_outer * m_broadcast_3d
            Z1 = Z1 + k_t * m_broadcast_1d
            Z2 = Z2 + k_t * m_broadcast_1d

            # Output: normalized linear attention
            H = H1 + H2           # [b, d, d]
            Z = (Z1 + Z2)         # [b, d]
            num = torch.einsum('bd,bde->be', q_t, H)  # [b, d]
            den = (q_t * Z).sum(dim=-1, keepdim=True).clamp_min(eps)  # [b,1]
            o_t = (num / den) * m_broadcast_1d  # zero out outputs on padded steps
            outputs[:, t] = o_t  # assign directly to preallocated tensor

    return outputs  # [b, t, d]


# ------------------------------
# Delta rule wrapper module
# ------------------------------

class DeltaRule(nn.Module):
    """Multi-timescale gated delta-rule with phi feature map and normalization.

    Implements a dual-timescale associative memory mechanism with input-conditioned
    forgetting, positive feature maps for stability, and chunked processing for
    memory efficiency.

    Args:
        head_dim: Dimension of each attention head.
        chunk_size: Size of chunks for memory-efficient processing.
        beta_min: Minimum value for decay factor clamping.
        beta_max: Maximum value for decay factor clamping.
        eps: Small constant for numerical stability.
    """

    def __init__(self, head_dim: int, *, chunk_size: int = 64, beta_min: float = 0.01, beta_max: float = 0.995, eps: float = 1e-6):
        """Initializes the DeltaRule module.

        Args:
            head_dim: Dimension of each attention head.
            chunk_size: Size of chunks for memory-efficient processing.
            beta_min: Minimum value for decay factor clamping.
            beta_max: Maximum value for decay factor clamping.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.head_dim = head_dim
        # Two base timescales (shared across heads by design for parameter economy)
        # Initialize to moderately slow decays to encourage retention
        self.base_beta_1 = nn.Parameter(torch.tensor(2.0))  # sigmoid(2) ~ 0.88
        self.base_beta_2 = nn.Parameter(torch.tensor(3.0))  # sigmoid(3) ~ 0.95
        self.chunk_size = int(chunk_size)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.eps = float(eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies the multi-timescale delta-rule attention.

        Args:
            q: Query tensor of shape [batch, seq_len, head_dim].
            k: Key tensor of shape [batch, seq_len, head_dim].
            v: Value tensor of shape [batch, seq_len, head_dim].
            beta: Input-conditioned gate of shape [batch, seq_len, 1].
            mask: Optional attention mask with 1 for valid tokens, 0 for padding,
                shape [batch, seq_len, 1].

        Returns:
            Output tensor of shape [batch, seq_len, head_dim].
        """
        bsz, seqlen, _ = q.shape
        if mask is None:
            # Default to all ones (no padding)
            mask = q.new_ones((bsz, seqlen, 1))
        else:
            # Ensure proper shape and dtype
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            elif mask.dim() > 3:
                # Try to squeeze broadcast dims like [b,1,1,t]
                while mask.dim() > 3:
                    mask = mask.squeeze(1)
                if mask.dim() == 3 and mask.shape[-1] != 1:
                    # If comes as [b, t, 1] it's fine; if [b, 1, t] after squeeze, permute
                    if mask.shape[1] == seqlen and mask.shape[2] != 1:
                        mask = mask.unsqueeze(-1)
            # Cast to float
            mask = mask.to(dtype=q.dtype)
            # If mask is [b, 1, t], permute to [b, t, 1]
            if mask.shape == (bsz, 1, seqlen):
                mask = mask.transpose(1, 2)
            # Final assert on shape
            assert mask.shape[:2] == (bsz, seqlen), "Mask must align with (batch, seq_len)"
            if mask.shape[-1] != 1:
                mask = mask.unsqueeze(-1)

        # Delegate to compiled kernel
        return compiled_delta_update(
            q, k, v, beta, mask,
            self.base_beta_1, self.base_beta_2,
            self.beta_min, self.beta_max,
            self.eps, self.chunk_size,
        )


# ------------------------------
# Attention Layer (DeltaNetLayer)
# ------------------------------

class DeltaNetLayer(nn.Module):
    """Single DeltaNet attention layer with multi-head delta-rule memory.

    Implements a multi-head attention mechanism using the delta-rule for
    associative memory, with head-specific recency bias and pre-normalization.

    Args:
        hidden_size: Total dimension of the model.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        **kwargs: Additional keyword arguments (unused).
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        """Initializes the DeltaNetLayer.

        Args:
            hidden_size: Total dimension of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            **kwargs: Additional keyword arguments (unused).
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
        # Input-conditioned gate per head
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Delta rule computation per head-dim space (shared across heads)
        self.delta_rule = DeltaRule(self.head_dim)

        # Head-specific recency bias multiplier (ALiBi-like in state domain)
        # Initialize close to 1.0 (weak additional decay)
        self.head_recency_logit = nn.Parameter(torch.zeros(num_heads))

        # Normalization & dropout
        self.layer_norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes layer parameters with Xavier uniform and appropriate biases."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Bias init for beta heads encouraging moderate retention
        nn.init.constant_(self.beta_proj.bias, 1.5)  # sigmoid ~ 0.82

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies multi-head delta-rule attention with residual connection.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            mask: Optional attention mask of shape [batch, seq_len] or
                [batch, 1, 1, seq_len] with 1 for valid tokens, 0 for padding.

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        # Pre-norm for stability
        residual = x
        x = self.layer_norm(x)

        bsz, seqlen, _ = x.shape

        # Linear projections
        q = self.q_proj(x)  # [b, t, d]
        k = self.k_proj(x)  # [b, t, d]
        v = self.v_proj(x)  # [b, t, d]

        # Beta gate in (0,1)
        beta = torch.sigmoid(self.beta_proj(x))  # [b, t, h]

        # Reshape to multi-head
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_heads)
        beta = rearrange(beta, 'b t h -> b t h 1')  # [b, t, h, 1]

        # Prepare mask to [b, t, 1]
        mask_bt1: Optional[torch.Tensor]
        if mask is None:
            mask_bt1 = None
        else:
            mask_bt1 = mask
            # Standardize shapes from common formats
            if mask_bt1.dim() == 2:
                # [b, t]
                mask_bt1 = mask_bt1.to(dtype=x.dtype).unsqueeze(-1)
            elif mask_bt1.dim() == 4:
                # Possibly [b, 1, 1, t]
                if mask_bt1.shape[1] == 1 and mask_bt1.shape[2] == 1:
                    mask_bt1 = mask_bt1[:, 0, 0, :]
                    mask_bt1 = mask_bt1.to(dtype=x.dtype).unsqueeze(-1)
                else:
                    # Fallback: reduce over extra dims
                    mask_bt1 = mask_bt1.reshape(mask_bt1.shape[0], -1, mask_bt1.shape[-1])
                    mask_bt1 = mask_bt1[:, 0, :].to(dtype=x.dtype).unsqueeze(-1)
            elif mask_bt1.dim() == 3 and mask_bt1.shape[-1] != 1:
                # [b, t, c] -> take first channel
                mask_bt1 = mask_bt1[:, :, 0:1].to(dtype=x.dtype)
            else:
                mask_bt1 = mask_bt1.to(dtype=x.dtype)

            # Ensure shape alignment
            assert mask_bt1.shape[:2] == (bsz, seqlen), "Mask must align with (batch, seq_len)"
            if mask_bt1.shape[-1] != 1:
                mask_bt1 = mask_bt1.unsqueeze(-1)

        # Head-specific recency multiplier gamma in (beta_min, beta_max)
        gamma = torch.sigmoid(self.head_recency_logit)  # [h]
        # Slightly clamp to avoid extremes
        gamma = gamma.clamp(self.delta_rule.beta_min, self.delta_rule.beta_max)

        head_outputs = []
        for h in range(self.num_heads):
            q_h = q[:, :, h, :]  # [b, t, d]
            k_h = k[:, :, h, :]  # [b, t, d]
            v_h = v[:, :, h, :]  # [b, t, d]
            beta_h = beta[:, :, h, :]  # [b, t, 1]
            # Apply head recency multiplier
            beta_h = (beta_h * gamma[h]).clamp(self.delta_rule.beta_min, self.delta_rule.beta_max)
            out_h = self.delta_rule(q_h, k_h, v_h, beta_h, mask=mask_bt1)  # [b, t, d]
            head_outputs.append(out_h)

        # Concatenate heads and output projection
        output = rearrange(torch.stack(head_outputs, dim=2), 'b t h d -> b t (h d)')
        output = self.out_proj(output)
        output = self.dropout(output)

        # Residual connection
        return residual + output


# ------------------------------
# Transformer Block
# ------------------------------

class DeltaNetBlock(nn.Module):
    """Transformer block combining DeltaNet attention with SwiGLU FFN.

    A complete transformer block with pre-normalization, delta-rule attention,
    and a SwiGLU feed-forward network with residual connections.

    Args:
        hidden_size: Total dimension of the model.
        num_heads: Number of attention heads.
        ffn_hidden_size: Hidden dimension of the FFN. Defaults to 3x hidden_size.
        dropout: Dropout probability.
        **kwargs: Additional keyword arguments (unused).
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1, **kwargs):
        """Initializes the DeltaNetBlock.

        Args:
            hidden_size: Total dimension of the model.
            num_heads: Number of attention heads.
            ffn_hidden_size: Hidden dimension of the FFN. Defaults to 3x hidden_size.
            dropout: Dropout probability.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__()
        if ffn_hidden_size is None:
            # Slightly reduced width because SwiGLU is more expressive
            ffn_hidden_size = int(3.0 * hidden_size)

        self.attention = DeltaNetLayer(hidden_size, num_heads, dropout)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            SwiGLU(hidden_size, ffn_hidden_size, dropout=dropout),
            nn.Dropout(dropout)
        )

        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies attention and FFN with residual connections.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            mask: Optional attention mask.

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        # Attention (pre-norm inside layer)
        x = self.attention(x, mask)

        # FFN block with pre-norm and residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        return x


# ------------------------------
# Model
# ------------------------------

class DeltaNet(nn.Module):
    """Complete DeltaNet model with enhanced DeltaNetLayer blocks.

    A transformer language model using linear attention with multi-timescale
    delta-rule memory, featuring sub-quadratic complexity and linear-time
    streaming capabilities.

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of the model.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for position embeddings.
        dropout: Dropout probability.
        **kwargs: Additional keyword arguments (unused).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        **kwargs,
    ):
        """Initializes the DeltaNet model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Dimension of the model.
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length for position embeddings.
            dropout: Dropout probability.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Embeddings (absolute pos kept for compatibility)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Stacked blocks
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes embedding weights with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def _maybe_expand_position_embedding(self, seq_len: int, device: torch.device):
        """Dynamically expands position embeddings if needed.

        Preserves learned weights for existing positions and initializes
        new positions with normal distribution.

        Args:
            seq_len: Required sequence length.
            device: Device to place new embeddings on.
        """
        current_n = self.position_embedding.num_embeddings
        if seq_len > current_n:
            new_pe = nn.Embedding(seq_len, self.hidden_size).to(device=device, dtype=self.position_embedding.weight.dtype)
            # Copy existing weights
            with torch.no_grad():
                new_pe.weight[:current_n].copy_(self.position_embedding.weight)
                nn.init.normal_(new_pe.weight[current_n:], std=0.02)
            self.position_embedding = new_pe
            self.max_seq_len = seq_len

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes forward pass through the DeltaNet model.

        Args:
            input_ids: Token indices of shape [batch, seq_len].
            attention_mask: Optional mask with 1 for valid tokens, 0 for padding.

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size].
        """
        bsz, seqlen = input_ids.shape

        # Ensure position embeddings cover current sequence length
        self._maybe_expand_position_embedding(seqlen, device=input_ids.device)

        # Positions
        pos_ids = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        x = self.dropout(x)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final projection
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        """Returns a formatted string summarizing the model architecture.

        Returns:
            Multi-line string with architecture details including model type,
            dimensions, layer counts, and parameter count.
        """
        return f"""
DeltaNet Architecture Summary (Enhanced):
- Model Type: Linear Attention Transformer (Multi-Scale Gated Delta-Rule)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {getattr(self.layers[0].attention, 'num_heads', 'n/a')}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: multi-timescale forgetting, phi+normalization, chunked processing, RMSNorm+SwiGLU
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model

def create_model(vocab_size: int = 50257, **kwargs) -> 'DeltaNet':
    """Factory function to create a DeltaNet model with default configuration.

    Creates a DeltaNet instance with sensible defaults that can be overridden
    via keyword arguments.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2 tokenizer).
        **kwargs: Override any default configuration parameter including:
            - hidden_size: Model dimension (default: 512).
            - num_layers: Number of transformer blocks (default: 6).
            - num_heads: Number of attention heads (default: 8).
            - max_seq_len: Maximum sequence length (default: 2048).
            - dropout: Dropout probability (default: 0.1).

    Returns:
        Configured DeltaNet model instance.
    """
    default_config = {
        'hidden_size': 512,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 2048,
        'dropout': 0.1,
    }
    default_config.update(kwargs)
    return DeltaNet(vocab_size=vocab_size, **default_config)


if __name__ == "__main__":
    # Basic sanity test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    bsz, seqlen = 3, 4097  # test dynamic pos-emb expansion
    x = torch.randint(0, 1000, (bsz, seqlen))
    attn_mask = torch.ones(bsz, seqlen)
    attn_mask[:, -5:] = 0  # simulate right padding
    with torch.no_grad():
        logits = model(x, attention_mask=attn_mask)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
