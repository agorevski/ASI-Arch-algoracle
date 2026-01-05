"""
DeltaNet: Normalized Multi-Timescale Linear Attention (Evolved)
This implementation evolves the seed architecture with:
- Normalized linear attention (running numerator/denominator)
- Multi-timescale forgetting per head (J memories) with adaptive mixing
- Content-gated memory updates
- ALiBi-inspired monotonic head-wise decay integrated into the recurrence
- Chunkwise causal processing for O(N) complexity
- einops.rearrange exclusively for reshaping
- @torch.compile on core scan kernel for performance
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes as described in Press et al.

    Generates monotonically decreasing slopes for each attention head,
    used to create position-dependent decay factors in the attention mechanism.

    Args:
        n_heads: Number of attention heads.

    Returns:
        Tensor of shape [n_heads] containing the computed slopes.
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = _get_alibi_slopes(2 * closest_power_of_2)  # type: ignore
        slopes += extra.tolist()[: n_heads - closest_power_of_2]
    return torch.tensor(slopes, dtype=torch.float32)


class DeltaNet(nn.Module):
    """Single evolved DeltaNet attention layer with normalized, multi-timescale linear attention.

    Forward signature preserved: forward(x: Tensor, mask: Optional[Tensor] = None) -> Tensor
    """

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        # New defaults (enabled by default without config changes)
        num_memories: int = 2,  # J multi-timescale memories per head
        chunk_size: int = 128,
        eps: float = 1e-6,
        # Compatibility alias for external configs
        d_model: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the DeltaNet attention layer.

        Args:
            hidden_size: Model hidden dimension. Either this or d_model must be provided.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            num_memories: Number of multi-timescale memories per head (J).
            chunk_size: Size of chunks for chunked causal processing.
            eps: Small epsilon for numerical stability in normalization.
            d_model: Alias for hidden_size for compatibility with external configs.
            **kwargs: Additional keyword arguments (ignored).

        Raises:
            ValueError: If neither hidden_size nor d_model is provided.
            AssertionError: If hidden_size is not divisible by num_heads.
        """
        super().__init__()
        # Allow using either hidden_size or d_model (standard name)
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNet: either hidden_size or d_model must be provided")
        if hidden_size is None:
            hidden_size = d_model  # type: ignore[assignment]
        self.hidden_size = int(hidden_size)  # type: ignore[arg-type]

        self.num_heads = num_heads
        self.head_dim = self.hidden_size // num_heads
        assert self.head_dim * num_heads == self.hidden_size, (
            f"hidden_size {self.hidden_size} not divisible by num_heads {num_heads}"
        )
        self.num_memories = num_memories
        self.chunk_size = chunk_size
        self.eps = eps

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Forgetting parameters (per head, per memory) and mixture weights
        self.beta_proj = nn.Linear(self.hidden_size, num_heads * num_memories, bias=True)
        self.mix_proj = nn.Linear(self.hidden_size, num_heads * num_memories, bias=True)

        # Content gate per head
        self.gate_proj = nn.Linear(self.hidden_size, num_heads, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        # ALiBi-inspired monotonic decay per head (register as buffer)
        slopes = _get_alibi_slopes(num_heads)
        # Convert to multiplicative decay r = exp(-slope) in (0,1)
        r = torch.exp(-slopes)
        # Store as [1, H, 1] to broadcast with [B, H, J]
        self.register_buffer("alibi_r", rearrange(r, "h -> 1 h 1"))  # [1,H,1]

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize layer parameters.

        Applies Xavier uniform initialization to projection weights,
        zeros to output bias, and specific values to beta/mix/gate biases
        for optimal initial behavior.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Initialize beta bias to favor retention (near 0.9 after sigmoid)
        nn.init.constant_(self.beta_proj.bias, 2.1972)  # sigmoid(2.1972) ≈ 0.9
        # Small biases for mix and gate
        nn.init.zeros_(self.mix_proj.bias)
        nn.init.zeros_(self.gate_proj.bias)

    @torch.compile
    def _process_chunk(
        self,
        q_chunk: torch.Tensor,  # [B, L, H, D]
        k_chunk: torch.Tensor,  # [B, L, H, D]
        v_chunk: torch.Tensor,  # [B, L, H, D]
        beta_chunk: torch.Tensor,  # [B, L, H, J]
        mix_chunk: torch.Tensor,  # [B, L, H, J]
        gate_chunk: torch.Tensor,  # [B, L, H, 1]
        mask_chunk: Optional[torch.Tensor],  # [B, L, 1, 1] or None
        S_state: torch.Tensor,  # [B, H, J, D, D] (float32)
        z_state: torch.Tensor,  # [B, H, J, D] (float32)
    ):
        """Process a single chunk with normalized multi-timescale linear attention.

        Performs causal attention computation within a chunk, updating the running
        numerator (S_state) and denominator (z_state) states for each timestep.

        Args:
            q_chunk: Query tensor of shape [B, L, H, D].
            k_chunk: Key tensor of shape [B, L, H, D].
            v_chunk: Value tensor of shape [B, L, H, D].
            beta_chunk: Forgetting parameter tensor of shape [B, L, H, J].
            mix_chunk: Memory mixture weights of shape [B, L, H, J].
            gate_chunk: Content gate tensor of shape [B, L, H, 1].
            mask_chunk: Optional attention mask of shape [B, L, 1, 1].
            S_state: Running numerator state of shape [B, H, J, D, D] in float32.
            z_state: Running denominator state of shape [B, H, J, D] in float32.

        Returns:
            Tuple of:
                - y_chunk: Output tensor of shape [B, L, H, D].
                - S_state: Updated numerator state of shape [B, H, J, D, D].
                - z_state: Updated denominator state of shape [B, H, J, D].
        """
        B, L, H, D = q_chunk.shape
        J = beta_chunk.shape[-1]
        eps = self.eps
        r = self.alibi_r  # [1,H,1]

        # Preallocate output buffer for better compilation/perf
        y_chunk = q_chunk.new_empty((B, L, H, D))

        for t in range(L):
            q_t = q_chunk[:, t]  # [B, H, D]
            k_t = k_chunk[:, t]  # [B, H, D]
            v_t = v_chunk[:, t]  # [B, H, D]
            beta_t = beta_chunk[:, t]  # [B, H, J]
            mix_t = mix_chunk[:, t]  # [B, H, J]
            g_t = gate_chunk[:, t]  # [B, H, 1]

            # Effective decay combines dynamic beta and monotonic head decay r
            beta_eff = torch.sigmoid(beta_t)  # [B,H,J]
            beta_eff = (beta_eff * r).clamp(max=0.9999)  # broadcast r -> [1,H,1]

            # Compute decay factors for states
            decay_S = beta_eff.unsqueeze(-1).unsqueeze(-1)  # [B,H,J,1,1]
            decay_z = beta_eff.unsqueeze(-1)  # [B,H,J,1]

            # Content-gated write terms
            outer = torch.einsum("bhd,bhe->bhde", k_t, v_t).to(torch.float32)  # [B,H,D,D]
            outer = outer.unsqueeze(2)  # [B,H,1,D,D]
            write_S = g_t.unsqueeze(-1).unsqueeze(-1) * outer  # [B,H,1,D,D] -> broadcast over J
            # Fix broadcasting by matching dims explicitly: [B,H,1,1] * [B,H,1,D]
            write_z = g_t.unsqueeze(-1) * k_t.unsqueeze(2)  # [B,H,1,D] -> broadcast over J

            if mask_chunk is not None:
                # Masked positions should not advance state (no decay, no write)
                m_t = mask_chunk[:, t]  # [B,1,1]
                m_S = m_t.to(S_state.dtype).unsqueeze(-1).unsqueeze(-1)  # [B,1,1,1,1]
                m_z = m_t.to(z_state.dtype).unsqueeze(-1)  # [B,1,1,1]

                S_state = S_state * (m_S * decay_S + (1.0 - m_S)) + m_S * write_S
                z_state = z_state * (m_z * decay_z + (1.0 - m_z)) + m_z * write_z
            else:
                # Unmasked case: always decay and write
                S_state = S_state * decay_S + write_S
                z_state = z_state * decay_z + write_z

            # Normalized retrieval per memory
            num = torch.einsum("bhd,bhjde->bhje", q_t, S_state)  # [B,H,J,D]
            denom = torch.einsum("bhd,bhjd->bhj", q_t, z_state) + eps  # [B,H,J]
            y_j = num / denom.unsqueeze(-1)  # [B,H,J,D]

            # Mixture weights across memories
            w = F.softmax(mix_t, dim=-1).unsqueeze(-1)  # [B,H,J,1]
            y = (y_j * w).sum(dim=2)  # [B,H,D]

            if mask_chunk is not None:
                y = y * mask_chunk[:, t]  # zero out outputs for padded positions

            y_chunk[:, t] = y

        return y_chunk, S_state, z_state

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform forward pass through the DeltaNet attention layer.

        Applies normalized multi-timescale linear attention with content gating
        and ALiBi-style decay, followed by output projection and layer normalization.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            mask: Optional attention mask of shape [batch, seq_len] where
                1 indicates valid positions and 0 indicates padding.

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        B, T, _ = x.shape

        residual = x

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Non-negative feature map for stability (approx. softmax kernel)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Reshape to multi-head: [B, T, H, D]
        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_heads)

        # Per-token dynamic parameters
        beta = self.beta_proj(x)  # [B,T,H*J]
        beta = rearrange(beta, "b t (h j) -> b t h j", h=self.num_heads, j=self.num_memories)
        mix = self.mix_proj(x)
        mix = rearrange(mix, "b t (h j) -> b t h j", h=self.num_heads, j=self.num_memories)
        gate = torch.sigmoid(self.gate_proj(x))  # [B,T,H]
        gate = rearrange(gate, "b t h -> b t h 1")

        # Prepare mask
        mask_chunk_base: Optional[torch.Tensor] = None
        if mask is not None:
            # [B,T] -> [B,T,1,1] for broadcast
            mask_chunk_base = rearrange(mask.to(q.dtype), "b t -> b t 1 1")

        # Initialize states (float32 accumulators for stability)
        D = self.head_dim
        H = self.num_heads
        J = self.num_memories
        S = torch.zeros((B, H, J, D, D), device=x.device, dtype=torch.float32)
        z = torch.zeros((B, H, J, D), device=x.device, dtype=torch.float32)

        # Chunked causal processing
        outputs = []
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            q_chunk = q[:, start:end]
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            beta_chunk = beta[:, start:end]
            mix_chunk = mix[:, start:end]
            gate_chunk = gate[:, start:end]
            mask_chunk = None if mask_chunk_base is None else mask_chunk_base[:, start:end]

            # Compute chunk outputs and update states
            y_chunk, S, z = self._process_chunk(
                q_chunk, k_chunk, v_chunk, beta_chunk, mix_chunk, gate_chunk, mask_chunk, S, z
            )
            outputs.append(y_chunk)

        y = torch.cat(outputs, dim=1)  # [B,T,H,D]
        y = rearrange(y, "b t h d -> b t (h d)")

        # Output projection, dropout, residual, layer norm
        y = self.out_proj(y)
        y = self.dropout(y)
        y = self.layer_norm(residual + y)
        return y


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        num_heads: int = 8,
        ffn_hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        # Compatibility alias for external configs
        d_model: Optional[int] = None,
    ):
        """Initialize a DeltaNet transformer block.

        Args:
            hidden_size: Model hidden dimension. Either this or d_model must be provided.
            num_heads: Number of attention heads.
            ffn_hidden_size: Hidden dimension of feed-forward network.
                Defaults to 4 * hidden_size if not provided.
            dropout: Dropout probability.
            d_model: Alias for hidden_size for compatibility with external configs.

        Raises:
            ValueError: If neither hidden_size nor d_model is provided.
        """
        super().__init__()

        # Allow either hidden_size or d_model
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNetBlock: either hidden_size or d_model must be provided")
        if hidden_size is None:
            hidden_size = d_model  # type: ignore[assignment]

        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * int(hidden_size)  # type: ignore[arg-type]

        self.attention = DeltaNet(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(int(hidden_size), ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, int(hidden_size)),
            nn.Dropout(dropout),
        )

        self.ffn_layer_norm = nn.LayerNorm(int(hidden_size))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform forward pass through the transformer block.

        Applies DeltaNet attention followed by a feed-forward network,
        each with residual connections and layer normalization.

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


class DeltaNetModel(nn.Module):
    """Complete DeltaNet language model with embeddings and stacked transformer blocks."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: Optional[int] = None,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        # Compatibility alias for external configs
        d_model: Optional[int] = None,
    ):
        """Initialize the DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Model hidden dimension. Either this or d_model must be provided.
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads per layer.
            max_seq_len: Maximum sequence length for positional embeddings.
            dropout: Dropout probability.
            d_model: Alias for hidden_size for compatibility with external configs.

        Raises:
            ValueError: If neither hidden_size nor d_model is provided.
        """
        super().__init__()

        # Allow either hidden_size or d_model
        if hidden_size is None and d_model is None:
            raise ValueError("DeltaNetModel: either hidden_size or d_model must be provided")
        if hidden_size is None:
            hidden_size = d_model  # type: ignore[assignment]

        self.hidden_size = int(hidden_size)  # type: ignore[arg-type]
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                DeltaNetBlock(hidden_size=self.hidden_size, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Output layer
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding parameters with normal distribution."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform forward pass through the complete DeltaNet model.

        Computes token and position embeddings, passes through all transformer
        blocks, and produces logits over the vocabulary.

        Args:
            input_ids: Input token IDs of shape [batch, seq_len].
            attention_mask: Optional attention mask of shape [batch, seq_len]
                where 1 indicates valid positions and 0 indicates padding.

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size].
        """
        B, T = input_ids.shape

        # Position IDs
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

        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        """Generate a human-readable summary of the model architecture.

        Returns:
            Multi-line string describing model type, dimensions,
            key innovations, and parameter count.
        """
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Normalized Multi-Timescale Linear Attention Transformer
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovations: Normalized delta update, multi-timescale decay, content gating, ALiBi-style decay, chunked O(N)
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """
    Create a DeltaNet model with default parameters

    Args:
        vocab_size: Vocabulary size
        **kwargs: Additional model parameters

    Returns:
        DeltaNetModel instance
    """
    default_config = {
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "max_seq_len": 2048,
        "dropout": 0.1,
    }
    default_config.update(kwargs)

    return DeltaNetModel(vocab_size=vocab_size, **default_config)


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8)
    batch_size, seq_len = 3, 97
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
