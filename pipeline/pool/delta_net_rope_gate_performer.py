"""
DeltaNet: Evolved Linear Attention Architecture with Performer-style normalization,
input-conditioned gating, RoPE positions, chunked processing, and pre-norm.
This evolution maintains sub-quadratic complexity while improving selectivity and stability.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Optional @torch.compile decorator helper
if hasattr(torch, 'compile'):
    def _compile_decorator(fn):
        return torch.compile(fn, fullgraph=False)
else:
    def _compile_decorator(fn):
        return fn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, dim]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def _get_rope_cache(seq_len: int, dim: int, base: float = 10000.0, device=None, dtype=None):
    # Returns cos, sin of shape [seq_len, dim//2]
    # Compute in float32 for numerical stability and to ensure CPU support for trig ops
    half = dim // 2
    compute_dtype = torch.float32
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=compute_dtype) / float(half)))
    t = torch.arange(seq_len, device=device, dtype=compute_dtype)
    freqs = torch.einsum('l,d->ld', t, inv_freq)  # [L, half]
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, L, D], D must be even
    B, H, L, D = x.shape
    half = D // 2
    x_even = x[..., 0:half]
    x_odd = x[..., half:]
    # cos/sin: [L, half] -> [1,1,L,half]
    cos = cos.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    # Apply rotation
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    return torch.cat([x_rot_even, x_rot_odd], dim=-1)


class DeltaNet(nn.Module):
    """Single DeltaNet attention layer with normalized linear attention and input-conditioned gating.
    - Performer-style positive feature maps with explicit normalization.
    - Per-head input-conditioned forget gates.
    - RoPE positional embedding on q/k.
    - Chunked causal processing with fp32 accumulators.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1,
                 chunk_size: int = 128, rope_base: float = 10000.0, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.rope_base = rope_base
        self.eps = eps

        assert self.head_dim * num_heads == hidden_size, \
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        assert (self.head_dim % 2) == 0, "head_dim must be even for RoPE"

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._reset_parameters()

        # Gate clamping bounds and base bias toward long memory
        beta0 = 0.98
        self.register_buffer('beta_min', torch.tensor(0.80), persistent=False)
        self.register_buffer('beta_max', torch.tensor(0.9995), persistent=False)
        # Bias is inside gate_proj.bias; initialize towards logit(beta0)
        with torch.no_grad():
            if self.gate_proj.bias is not None:
                self.gate_proj.bias.fill_(math.log(beta0 / (1 - beta0)))

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @_compile_decorator
    def _process_chunk(self,
                       q: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor,
                       beta: torch.Tensor,
                       attn_mask: Optional[torch.Tensor],
                       S: torch.Tensor,
                       z: torch.Tensor):
        """
        Process a chunk sequentially in a compiled function.
        Args:
            q,k,v: [B, H, T, D]
            beta: [B, H, T] in (0,1)
            attn_mask: Optional [B, T] with 1 for valid, 0 for pad
            S: state numerator matrix [B, H, D, D] (fp32)
            z: state denominator vector [B, H, D] (fp32)
        Returns:
            y: [B, H, T, D], S, z (updated states)
        """
        B, H, T, D = q.shape
        y_out = []
        # Prepare mask tensors
        if attn_mask is not None:
            # [B,T]
            m_tok = attn_mask.to(dtype=q.dtype)
        else:
            m_tok = None

        for t in range(T):
            q_t = q[:, :, t, :]  # [B,H,D]
            k_t = k[:, :, t, :]  # [B,H,D]
            v_t = v[:, :, t, :]  # [B,H,D]
            beta_t = beta[:, :, t]  # [B,H]

            if m_tok is not None:
                # Scalars per batch for time t
                m_scalar = m_tok[:, t].unsqueeze(1)  # [B,1]
                # Effective beta: if m=1 use beta_t else 1.0 (no decay across padding)
                eff_beta_t = beta_t * m_scalar + (1.0 - m_scalar)  # [B,H]
                # Broadcast scale across heads and dim
                scale_t = m_scalar.unsqueeze(-1)  # [B,1,1]
                k_t = k_t * scale_t
                v_t = v_t * scale_t
                q_scale = scale_t
            else:
                eff_beta_t = beta_t
                q_scale = None

            # Decay previous states
            S = S * eff_beta_t.unsqueeze(-1).unsqueeze(-1)  # [B,H,D,D]
            z = z * eff_beta_t.unsqueeze(-1)  # [B,H,D]

            # Positive feature maps
            phi_k = F.elu(k_t, alpha=1.0) + 1.0  # [B,H,D]
            phi_q = F.elu(q_t, alpha=1.0) + 1.0  # [B,H,D]

            # Accumulate numerator and denominator in fp32
            outer = torch.einsum('bhd,bhe->bhde', phi_k.to(dtype=S.dtype), v_t.to(dtype=S.dtype))
            S = S + outer
            if m_tok is not None:
                # Do not update denominator on padding tokens
                z = z + (phi_k.to(dtype=z.dtype) * scale_t)
            else:
                z = z + phi_k.to(dtype=z.dtype)

            # Output: y_t = (phi_q^T S) / (phi_q^T z + eps)
            if q_scale is not None:
                phi_q_for_out = phi_q * q_scale
            else:
                phi_q_for_out = phi_q
            numer = torch.einsum('bhd,bhde->bhe', phi_q_for_out.to(dtype=S.dtype), S)
            denom = (phi_q.to(dtype=z.dtype) * z).sum(dim=-1, keepdim=True) + self.eps
            # Compute the division in fp32 for stability, then cast back to input dtype
            y_t = (numer / denom).to(dtype=q.dtype)
            y_out.append(y_t.unsqueeze(2))  # [B,H,1,D]

        return torch.cat(y_out, dim=2), S, z

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, hidden_size]
            mask: Optional [B, L] attention mask (1 for valid tokens, 0 for padding)
        Returns:
            [B, L, hidden_size]
        """
        B, L, _ = x.shape

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Rearrange to [B,H,L,D]
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # RoPE on q,k
        cos, sin = _get_rope_cache(L, self.head_dim, base=self.rope_base, device=x.device, dtype=q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Input-conditioned gate beta in (0,1), per head per token
        beta_logits = self.gate_proj(x)  # [B,L,H]
        beta = torch.sigmoid(beta_logits)
        # Clamp to [beta_min, beta_max]
        beta = beta.clamp(min=float(self.beta_min.item()), max=float(self.beta_max.item()))
        beta = rearrange(beta, 'b l h -> b h l')  # [B,H,L]

        # Chunked processing with fp32 accumulators
        S = torch.zeros((B, self.num_heads, self.head_dim, self.head_dim), device=x.device, dtype=torch.float32)
        z = torch.zeros((B, self.num_heads, self.head_dim), device=x.device, dtype=torch.float32)

        outputs = []
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, start:end, :]
            v_chunk = v[:, :, start:end, :]
            beta_chunk = beta[:, :, start:end]
            mask_chunk = mask[:, start:end] if mask is not None else None

            y_chunk, S, z = self._process_chunk(q_chunk, k_chunk, v_chunk, beta_chunk, mask_chunk, S, z)
            outputs.append(y_chunk)

        y = torch.cat(outputs, dim=2)  # [B,H,L,D]
        y = rearrange(y, 'b h l d -> b l (h d)')

        # Output projection and dropout
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeltaNetBlock(nn.Module):
    """Complete DeltaNet transformer block with pre-norm architecture."""

    def __init__(self, hidden_size: int, num_heads: int = 8,
        ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.attn = DeltaNet(hidden_size, num_heads, dropout)
        self.attn_norm = RMSNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_hidden_size, dropout)
        self.ffn_norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attn(self.attn_norm(x), mask)
        # Pre-norm FFN
        x = x + self.ffn(self.ffn_norm(x))
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
        # Absolute position embeddings removed to avoid fixed-length constraints; RoPE is applied in attention
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            DeltaNetBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = input_ids.shape

        # Embeddings (token only; positional information is injected via RoPE inside attention)
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final norm and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        return f"""
DeltaNet Architecture Summary:
- Model Type: Normalized Linear Attention Transformer (RoPE + input-conditioned gating)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attn.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Positional Encoding: RoPE on q/k (no absolute position embeddings)
- Key Innovations: Performer-style normalization, input-conditioned forget gates, RoPE, chunked fp32 scan, pre-norm
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function for creating the model
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
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
    # Basic smoke test
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8)
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
