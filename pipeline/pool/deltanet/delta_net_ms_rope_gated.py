"""
DeltaNet: Evolved Linear Attention with Multi-Scale Gated Delta Memory and RoPE
- Sub-quadratic (O(n)) causal sequence modeling
- Chunkwise processing for memory efficiency and better hardware utilization
- Vectorized multi-head, multi-timescale associative memory with gated updates
- Rotary Position Embeddings (RoPE) for strong relative position inductive bias
- Batch-size agnostic operations with einops.rearrange for all reshapes
- @torch.compile applied to core per-chunk kernel
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==========================
# Utility: Rotary Embeddings
# ==========================

def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    """Build RoPE cos/sin caches for a given sequence length and head dimension.

    Uses float32 for cache construction for numerical stability, then casts to dtype.

    Args:
        seq_len: Sequence length for position embeddings.
        head_dim: Dimension of each attention head (must be even).
        device: Torch device for tensor allocation.
        dtype: Target dtype for output tensors.

    Returns:
        Tuple of (cos, sin) tensors, each with shape [1, seq_len, 1, head_dim].

    Raises:
        AssertionError: If head_dim is not even.
    """
    assert head_dim % 2 == 0, "RoPE head_dim must be even"
    half_dim = head_dim // 2
    # Build in float32 to avoid overflow/precision issues under mixed precision
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
    # positions [seq_len, 1]
    t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = t * inv_freq.unsqueeze(0)  # [seq_len, half_dim]
    # Expand to full dim by duplicating for interleaved pairs
    # emb shape [seq_len, head_dim]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(2).to(dtype=dtype)  # [1, seq_len, 1, head_dim]
    sin = emb.sin().unsqueeze(0).unsqueeze(2).to(dtype=dtype)  # [1, seq_len, 1, head_dim]
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Interleaves even/odd dimensions and applies rotation using cos/sin caches.

    Args:
        x: Input tensor of shape [batch, seq_len, heads, head_dim].
        cos: Cosine cache of shape [1, seq_len, 1, head_dim].
        sin: Sine cache of shape [1, seq_len, 1, head_dim].

    Returns:
        Tensor with RoPE applied, same shape as input [batch, seq_len, heads, head_dim].
    """
    # Interleave even/odd dimensions
    x_1 = x[..., ::2]
    x_2 = x[..., 1::2]
    cos_half = cos[..., : x_1.shape[-1]]
    sin_half = sin[..., : x_1.shape[-1]]
    out_1 = x_1 * cos_half - x_2 * sin_half
    out_2 = x_1 * sin_half + x_2 * cos_half
    out = torch.stack((out_1, out_2), dim=-1)
    out = rearrange(out, "... d two -> ... (d two)")
    return out


# =============================================
# Core compiled per-chunk multi-scale delta rule
# =============================================

@torch.compile
def msdelta_chunk(
    q_chunk: torch.Tensor,  # [B, C, H, D]
    k_chunk: torch.Tensor,  # [B, C, H, D]
    v_chunk: torch.Tensor,  # [B, C, H, D]
    gate_f: torch.Tensor,   # [B, C, H]
    gate_u: torch.Tensor,   # [B, C, H]
    w_mix: torch.Tensor,    # [B, C, H, M]
    alphas: torch.Tensor,   # [H, M] in (0,1)
    state: torch.Tensor,    # [B, H, M, D, D]
    token_mask: Optional[torch.Tensor] = None,  # [B, C] or None
):
    """Process one chunk sequentially with multi-scale gated delta updates.

    Applies delta rule updates to multi-scale associative memory states
    with gated retention and update mechanisms.

    Args:
        q_chunk: Query tensor of shape [B, C, H, D] where B=batch, C=chunk_size,
            H=heads, D=head_dim.
        k_chunk: Key tensor of shape [B, C, H, D].
        v_chunk: Value tensor of shape [B, C, H, D].
        gate_f: Forget gate tensor of shape [B, C, H] controlling retention.
        gate_u: Update gate tensor of shape [B, C, H] controlling write strength.
        w_mix: Timescale mixing weights of shape [B, C, H, M] where M=num_scales.
        alphas: Base decay rates per head per scale of shape [H, M] in range (0, 1).
        state: Multi-scale memory state of shape [B, H, M, D, D].
        token_mask: Optional mask of shape [B, C] for padding tokens.

    Returns:
        Tuple of (out_chunk, new_state) where:
            - out_chunk: Output tensor of shape [B, C, H, D].
            - new_state: Updated memory state of shape [B, H, M, D, D].
    """
    B, C, H, D = q_chunk.shape
    M = w_mix.shape[-1]
    out_list = []
    # Prepare broadcast shapes
    # alphas_b: [1, H, M, 1, 1]
    alphas_b = alphas.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    for t in range(C):
        qt = q_chunk[:, t]            # [B, H, D]
        kt = k_chunk[:, t]            # [B, H, D]
        vt = v_chunk[:, t]            # [B, H, D]
        gf = gate_f[:, t]             # [B, H]
        gu = gate_u[:, t]             # [B, H]
        wt = w_mix[:, t]              # [B, H, M]

        # Optional token mask for padding: [B] -> broadcast
        if token_mask is not None:
            mt = token_mask[:, t].unsqueeze(-1)  # [B, 1]
        else:
            mt = None

        # Outer product per head: [B, H, D, D]
        outer = torch.einsum("bhd,bhe->bhde", kt, vt)

        # Effective retention per timescale: alpha(h,m) * gate_f(b,h)
        # gf_b: [B, H, 1, 1], expand for [B, H, M, 1, 1]
        gf_b = gf.unsqueeze(-1).unsqueeze(-1)
        retain = alphas_b * gf_b.unsqueeze(2)  # [B, H, M, 1, 1]

        # Write routing per timescale with update gate: [B, H, M]
        write_w = wt * gu.unsqueeze(-1)  # [B, H, M]
        write_term = outer.unsqueeze(2) * write_w.unsqueeze(-1).unsqueeze(-1)  # [B, H, M, D, D]

        # State update with proper masking semantics:
        # If mt == 0 (masked/padded), keep state unchanged (no decay, no write)
        # If mt == 1 (valid), apply decay (retain) and write
        if mt is not None:
            mt_state = mt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1, 1]
            state = state * (retain * mt_state + (1.0 - mt_state)) + write_term * mt_state
        else:
            state = state * retain + write_term  # [B, H, M, D, D]

        # Read: mixture over timescales
        S_read = (state * wt.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)  # [B, H, D, D]
        o_t = torch.einsum("bhd,bhde->bhe", qt, S_read)  # [B, H, D]
        if mt is not None:
            o_t = o_t * mt.unsqueeze(-1)
        out_list.append(o_t)

    out_chunk = torch.stack(out_list, dim=1)  # [B, C, H, D]
    return out_chunk, state


class DeltaNet(nn.Module):
    """DeltaNet attention layer with multi-timescale gated delta memory and RoPE.
    Forward signature preserved: forward(x, mask=None) -> [B, S, D]
    """

    def __init__(self, hidden_size: Optional[int] = None, num_heads: int = 8, dropout: float = 0.1, d_model: Optional[int] = None, **kwargs):
        """Initialize DeltaNet attention layer.

        Args:
            hidden_size: Model hidden dimension. Either this or d_model must be specified.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            d_model: Alias for hidden_size for compatibility.
            **kwargs: Additional arguments (ignored).

        Raises:
            AssertionError: If neither hidden_size nor d_model is specified,
                if hidden_size is not divisible by num_heads, or if head_dim is odd.
        """
        super().__init__()
        # Support both hidden_size and d_model for compatibility
        if hidden_size is None and d_model is not None:
            hidden_size = d_model
        assert hidden_size is not None, "DeltaNet requires hidden_size (or d_model) to be specified"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        )
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Multi-scale gating and routing
        self.num_scales = 2  # default enabled
        # gates: [gate_f, gate_u] per head; mix weights per head per scale
        self.gate_proj = nn.Linear(hidden_size, num_heads * (2 + self.num_scales), bias=True)

        # Learnable base decays per head per scale (sigmoid to (0,1))
        # Initialize to favor long and medium horizons
        init_decays = torch.tensor([0.95, 0.80], dtype=torch.float32)
        logit = torch.logit(init_decays.clamp(1e-4, 1 - 1e-4))  # [2]
        self.logit_alphas = nn.Parameter(logit.unsqueeze(0).repeat(num_heads, 1))  # [H, M]

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Chunking for sequence processing
        self.chunk_size = 128  # sensible default

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize layer parameters with Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.xavier_uniform_(self.gate_proj.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through DeltaNet attention layer.

        Args:
            x: Input tensor of shape [B, S, D] where B=batch, S=sequence_length,
                D=hidden_size.
            mask: Optional attention mask of shape [B, S] or [B, S, 1] where
                1 indicates valid tokens and 0 indicates padding.

        Returns:
            Output tensor of shape [B, S, D] after attention and residual connection.
        """
        B, S, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        residual = x

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head with einops
        q = rearrange(q, "b s (h d) -> b s h d", h=H)
        k = rearrange(k, "b s (h d) -> b s h d", h=H)
        v = rearrange(v, "b s (h d) -> b s h d", h=H)

        # RoPE caches and application
        cos, sin = _build_rope_cache(S, Hd, device=x.device, dtype=x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Gating and timescale routing
        gates_all = self.gate_proj(x)  # [B, S, H*(2+M)]
        gates_all = rearrange(gates_all, "b s (h g) -> b s h g", h=H)
        # Split into gates and mix
        g_f = torch.sigmoid(gates_all[..., 0])        # [B, S, H]
        g_u = torch.sigmoid(gates_all[..., 1])        # [B, S, H]
        w_mix_logits = gates_all[..., 2:]             # [B, S, H, M]
        w_mix = torch.softmax(w_mix_logits, dim=-1)   # [B, S, H, M]

        # Alphas per head per scale in (0,1)
        alphas = torch.sigmoid(self.logit_alphas)  # [H, M]

        # Prepare optional token mask to handle padding: [B, S]
        token_mask = None
        if mask is not None:
            # Accept [B, S] or [B, S, 1]; convert to float in {0,1}
            if mask.dim() == 3:
                token_mask = mask.squeeze(-1).to(dtype=x.dtype)
            else:
                token_mask = mask.to(dtype=x.dtype)
        
        # Initialize multi-scale states: [B, H, M, D, D]
        state = q.new_zeros((B, H, self.num_scales, Hd, Hd))

        outputs = []
        C = self.chunk_size
        for start in range(0, S, C):
            end = min(start + C, S)
            q_chunk = q[:, start:end]  # [B, c, H, D]
            k_chunk = k[:, start:end]
            v_chunk = v[:, start:end]
            gf_chunk = g_f[:, start:end]
            gu_chunk = g_u[:, start:end]
            w_chunk = w_mix[:, start:end]
            mask_chunk = None if token_mask is None else token_mask[:, start:end]

            out_chunk, state = msdelta_chunk(
                q_chunk, k_chunk, v_chunk,
                gf_chunk, gu_chunk, w_chunk,
                alphas, state, mask_chunk
            )
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=1)  # [B, S, H, D]
        out = rearrange(out, "b s h d -> b s (h d)")
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        return out


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block (attention + FFN)."""

    def __init__(self, hidden_size: Optional[int] = None, num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1, d_model: Optional[int] = None):
        """Initialize DeltaNet transformer block.

        Args:
            hidden_size: Model hidden dimension. Either this or d_model must be specified.
            num_heads: Number of attention heads.
            ffn_hidden_size: Feed-forward network hidden dimension.
                Defaults to 4 * hidden_size.
            dropout: Dropout probability.
            d_model: Alias for hidden_size for compatibility.

        Raises:
            AssertionError: If neither hidden_size nor d_model is specified.
        """
        super().__init__()
        # Support both hidden_size and d_model for compatibility
        if hidden_size is None and d_model is not None:
            hidden_size = d_model
        assert hidden_size is not None, "DeltaNetBlock requires hidden_size (or d_model) to be specified"
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size
        self.attention = DeltaNet(hidden_size, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through DeltaNet transformer block.

        Args:
            x: Input tensor of shape [B, S, D].
            mask: Optional attention mask of shape [B, S] or [B, S, 1].

        Returns:
            Output tensor of shape [B, S, D] after attention and FFN.
        """
        x = self.attention(x, mask)
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


class DeltaNetModel(nn.Module):
    """Complete DeltaNet model wrapper (token embedding + stacked DeltaNet blocks + LM head).
    Uses RoPE inside attention; no absolute positional embeddings.
    """

    def __init__(self, vocab_size: int, hidden_size: Optional[int] = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1, d_model: Optional[int] = None):
        """Initialize DeltaNet language model.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Model hidden dimension.
            num_layers: Number of DeltaNet transformer blocks.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
            d_model: Alias for hidden_size for compatibility.

        Raises:
            AssertionError: If neither hidden_size nor d_model is specified.
        """
        super().__init__()
        # Support both hidden_size and d_model for compatibility
        if hidden_size is None and d_model is not None:
            hidden_size = d_model
        elif d_model is not None and hidden_size is not None and d_model != hidden_size:
            # If both provided, prefer hidden_size but warn via assert message if mismatch
            pass
        assert hidden_size is not None, "DeltaNetModel requires hidden_size (or d_model) to be specified"
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        # Stacked layers
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding weights with normal distribution (std=0.02)."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through DeltaNet language model.

        Args:
            input_ids: Token IDs of shape [B, S] where B=batch, S=sequence_length.
            attention_mask: Optional mask of shape [B, S] for padding tokens.

        Returns:
            Logits tensor of shape [B, S, vocab_size].
        """
        B, S = input_ids.shape
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        """Generate a human-readable summary of the model architecture.

        Returns:
            Multi-line string describing model configuration and parameter count.
        """
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Linear Attention Transformer with Multi-Scale Gated Delta Memory
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Positional Encoding: RoPE applied to q/k
- Memory: {self.layers[0].attention.num_scales} timescales per head with learnable decays
- Chunk Size: {self.layers[0].attention.chunk_size}
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model

def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
    """Create a DeltaNet model with default parameters.

    Factory function for instantiating DeltaNetModel with sensible defaults.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 50257 (GPT-2 vocab size).
        **kwargs: Additional model parameters to override defaults. Supported keys:
            - hidden_size (int): Model hidden dimension. Default 512.
            - num_layers (int): Number of transformer blocks. Default 6.
            - num_heads (int): Number of attention heads. Default 8.
            - max_seq_len (int): Maximum sequence length. Default 2048.
            - dropout (float): Dropout probability. Default 0.1.

    Returns:
        Configured DeltaNetModel instance ready for training or inference.
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
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4)
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
