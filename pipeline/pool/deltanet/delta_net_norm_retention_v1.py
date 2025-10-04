"""
DeltaNet: Evolved Linear Attention with Normalized Retention and Dynamic Forgetting
This implementation upgrades the baseline DeltaNet with:
- Per-head dynamic forgetting gates (beta) conditioned on input
- Performer-style linear attention normalization via running denominator state
- Chunkwise causal processing for O(N) complexity and memory efficiency
- einops.rearrange for all shape manipulations (no .view/.reshape)
- Batch-size agnostic operations throughout
- Retains original model interfaces and factory function

Research inspirations:
- Katharopoulos et al. (Transformers are RNNs): linear attention with feature maps
- Performer (Choromanski et al.): softmax approximation with normalization denominator
- RetNet/Mamba/RWKV: dynamic retention/forgetting gates for long-range memory
"""

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -------------------------
# Utility Normalization
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, dim]
        norm_x = x.norm(dim=-1, keepdim=True)
        d = x.shape[-1]
        rms = norm_x / math.sqrt(d)
        x_hat = x / (rms + self.eps)
        return x_hat * self.weight


# -------------------------
# Activation: SwiGLU FFN
# -------------------------
class SwiGLU(nn.Module):
    """SwiGLU gating MLP producing an expanded hidden representation.

    Given input dim_in, produces dim_out via:
      a = W1 x, b = W2 x
      y = (silu(a) * b) -> Dropout -> Linear(dim_out -> dim_out)
    Typically followed by a projection back to model dim outside this module.
    """
    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out, bias=True)
        self.w2 = nn.Linear(dim_in, dim_out, bias=True)
        # Keep a light projection to preserve original structure, but with correct dims
        self.w3 = nn.Linear(dim_out, dim_out, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)  # [*, dim_out]
        b = self.w2(x)  # [*, dim_out]
        x = F.silu(a) * b  # [*, dim_out]
        x = self.w3(self.dropout(x))  # [*, dim_out]
        return x


# -------------------------
# Core DeltaNet Layer (Attention)
# -------------------------
class DeltaNet(nn.Module):
    """Single DeltaNet attention layer with normalized linear attention and dynamic retention.

    The layer computes for each head h, time t:
      S_t = beta_t .* S_{t-1} + phi(k_t) ⊗ v_t
      Z_t = beta_t .* Z_{t-1} + phi(k_t)
      y_t = (phi(q_t) @ S_t) / clamp(phi(q_t) @ Z_t, eps)
    where beta_t in (0,1) is a learned, input-conditioned forgetting gate.

    All operations are causal and processed chunkwise for memory efficiency.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        chunk_size: int = 128,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dynamic forgetting gate per head
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        # Optional ALiBi-like slope per head integrated into beta computation
        self.beta_slope = nn.Parameter(torch.zeros(num_heads))  # initialized to 0

        # Output
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Norm inside layer (kept for compatibility with baseline)
        self.layer_norm = RMSNorm(hidden_size)

        # Misc
        self.chunk_size = chunk_size
        self.eps = eps

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Initialize beta bias for long half-life (e.g., ~512 tokens)
        half_life = 512.0
        beta_init = 0.5 ** (1.0 / half_life)
        # map to sigmoid pre-activation
        b = math.log(beta_init / (1.0 - beta_init))
        nn.init.constant_(self.beta_proj.bias, b)
        nn.init.zeros_(self.beta_proj.weight)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        # Non-negative feature map for linear attention
        return F.elu(x) + 1.0

    @torch.compile(mode="default", fullgraph=False)
    def _chunkwise_linear_attention(
        self,
        q: torch.Tensor,  # [b, h, s, d]
        k: torch.Tensor,  # [b, h, s, d]
        v: torch.Tensor,  # [b, h, s, d]
        beta: torch.Tensor,  # [b, h, s]
        attn_mask: torch.Tensor,  # [b, s] (guaranteed non-None by caller)
    ) -> torch.Tensor:
        b, h, s, d = q.shape
        d_v = v.shape[-1]

        # Prepare outputs
        y = torch.empty((b, h, s, d_v), device=q.device, dtype=q.dtype)

        # States per batch and head
        S = torch.zeros((b, h, d, d_v), device=q.device, dtype=q.dtype)
        Z = torch.zeros((b, h, d), device=q.device, dtype=q.dtype)

        # Expand mask to [b, 1, s, 1] for values and [b, 1, s] for beta/keys
        valid = attn_mask.to(device=q.device, dtype=q.dtype)
        valid = valid[:, None, :, None]  # [b,1,s,1]
        valid_beta = valid.squeeze(-1)   # [b,1,s]

        chunk = self.chunk_size
        for start in range(0, s, chunk):
            end = min(start + chunk, s)
            q_c = q[:, :, start:end, :]  # [b,h,l,d]
            k_c = k[:, :, start:end, :]
            v_c = v[:, :, start:end, :]
            beta_c = beta[:, :, start:end]  # [b,h,l]
            l = end - start

            # Apply feature maps
            qf = self._phi(q_c)  # [b,h,l,d]
            kf = self._phi(k_c)  # [b,h,l,d]

            # Masks for this chunk
            valid_c = valid[:, :, start:end, :]  # [b,1,l,1]
            valid_beta_c = valid_beta[:, :, start:end]  # [b,1,l]

            # Iterate inside chunk to maintain strict causality
            for t in range(l):
                q_t = qf[:, :, t, :]  # [b,h,d]
                k_t = kf[:, :, t, :]  # [b,h,d]
                v_t = v_c[:, :, t, :]  # [b,h,d_v]
                beta_t = beta_c[:, :, t]  # [b,h]

                # Optional masking: if position is padding, skip update and skipping forgetting
                mask_t = valid_beta_c[:, :, t]  # [b,1]
                # Broadcast to heads dimension if needed (will broadcast in where as well)
                # For padded tokens, set beta_t=1 so states are unchanged (no forgetting)
                beta_t = torch.where(mask_t > 0.5, beta_t, torch.ones_like(beta_t))

                # Forgetting
                S = S * beta_t[:, :, None, None]
                Z = Z * beta_t[:, :, None]

                # Update states with current token
                m_t = valid_c[:, :, t, :]  # [b,1,1]
                k_t_eff = k_t * m_t  # zero out padded positions
                v_t_eff = v_t * m_t

                # Outer product accumulation: S += k_t ⊗ v_t
                # k_t: [b,h,d], v_t: [b,h,d_v] -> [b,h,d,d_v]
                S = S + torch.einsum('bhd,bhe->bhde', k_t_eff, v_t_eff)
                # Denominator accumulation
                Z = Z + k_t_eff

                # Compute output: y_t = (q_t @ S) / (q_t @ Z)
                num = torch.einsum('bhd,bhde->bhe', q_t, S)  # [b,h,d_v]
                den = torch.einsum('bhd,bhd->bh', q_t, Z)    # [b,h]
                den = den.unsqueeze(-1)  # [b,h,1]
                y[:, :, start + t, :] = num / (den + self.eps)

        return y

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Residual
        residual = x

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Shapes: [b, s, h, d]
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        # Dynamic beta per head, per position (sigmoid + clamping)
        beta_logits = self.beta_proj(x)  # [b, s, h]
        # Add ALiBi-like slope term: slope_h * (t / S)
        bsz, seqlen = x.shape[0], x.shape[1]
        t_idx = torch.arange(seqlen, device=x.device, dtype=x.dtype)
        if seqlen > 1:
            tau = t_idx / (seqlen - 1)
        else:
            tau = t_idx
        tau = rearrange(tau, 's -> 1 s 1')  # [1,s,1]
        slope = rearrange(self.beta_slope, 'h -> 1 1 h')  # [1,1,h]
        beta_logits = beta_logits + slope * tau  # broadcast

        beta = torch.sigmoid(beta_logits)  # [b, s, h]
        # clamp to stable range to avoid vanishing or exploding memory
        beta = beta.clamp(0.90, 0.9995)
        beta = rearrange(beta, 'b s h -> b h s')

        # Prepare mask: always pass a tensor to avoid graph recompiles with torch.compile
        if mask is None:
            attn_mask = torch.ones(bsz, seqlen, device=x.device, dtype=x.dtype)
        else:
            attn_mask = mask.to(device=x.device)

        # Chunkwise normalized linear attention
        y = self._chunkwise_linear_attention(q, k, v, beta, attn_mask)  # [b,h,s,d]

        # Merge heads
        y = rearrange(y, 'b h s d -> b s (h d)')

        # Output projection, dropout, residual + norm
        out = self.out_proj(y)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        return out


# -------------------------
# Transformer Block
# -------------------------
class DeltaNetBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        ffn_hidden_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size

        self.attention = DeltaNet(hidden_size, num_heads, dropout)

        # Feed-forward network (SwiGLU)
        self.ffn = nn.Sequential(
            SwiGLU(hidden_size, ffn_hidden_size, dropout=dropout),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_layer_norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (already residual + norm inside)
        x = self.attention(x, mask)
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


# -------------------------
# Model
# -------------------------
class DeltaNetModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
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
        self.layer_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tie weights

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s = input_ids.shape
        # Position IDs
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0)
        pos = pos.expand(b, -1)

        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.dropout(x)

        # Layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final norm + LM head
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def get_architecture_summary(self) -> str:
        return f"""
DeltaNet Architecture Summary (Evolved):
- Model Type: Normalized Linear Attention (DeltaNet)
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Innovations: Dynamic per-head forgetting, normalized retention, chunkwise processing, RMSNorm, SwiGLU
- Parameters: {sum(p.numel() for p in self.parameters()):,}
"""


# Factory function
def create_model(vocab_size: int = 50257, **kwargs) -> DeltaNetModel:
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
    torch.manual_seed(0)
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4)
    bsz, seqlen = 2, 100
    input_ids = torch.randint(0, 1000, (bsz, seqlen))
    attn_mask = torch.ones(bsz, seqlen, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, attn_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
