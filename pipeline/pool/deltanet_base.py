"""
DeltaNet: Baseline Linear Attention Architecture
This is the vanilla seed architecture for ASI-Arch experiments.
Based on the DeltaNet paper: https://arxiv.org/abs/2406.06484
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DeltaRule(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = q.shape
        # Initialize state
        h_t = torch.zeros(batch_size, hidden_size, hidden_size, device=q.device, dtype=q.dtype)
        outputs = []
        # Sequential computation
        for t in range(seq_len):
            q_t = q[:, t:t+1]  # [batch, 1, hidden_size]
            k_t = k[:, t:t+1]  # [batch, 1, hidden_size]
            v_t = v[:, t:t+1]  # [batch, 1, hidden_size]
            beta_t = beta[:, t:t+1]  # [batch, 1, 1]
            # Apply forgetting mechanism - fix dimension issues
            beta_expanded = beta_t.squeeze(-1).unsqueeze(-1)  # [batch, 1, 1]
            h_t = h_t * beta_expanded
            # Update with outer product - fix einsum
            # k_t: [batch, 1, hidden_size], v_t: [batch, 1, hidden_size]
            # Want: [batch, hidden_size, hidden_size]
            outer_product = torch.einsum('bik,bji->bjk', k_t, v_t)
            h_t = h_t + outer_product
            # Compute output - fix einsum for proper dimensions
            # q_t: [batch, 1, hidden_size], h_t: [batch, hidden_size, hidden_size]
            # Want: [batch, 1, hidden_size]
            o_t = torch.einsum('bij,bjk->bik', q_t, h_t)
            outputs.append(o_t)
        return torch.cat(outputs, dim=1)


class DeltaNetLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, \
            f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # Delta rule computation
        self.delta_rule = DeltaRule(self.head_dim)
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Initialize beta to small positive values
        nn.init.constant_(self.beta_proj.bias, 0.9)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        # Store residual
        residual = x
        # Linear projections
        q = self.q_proj(x)  # [batch, seq_len, hidden_size]
        k = self.k_proj(x)  # [batch, seq_len, hidden_size]
        v = self.v_proj(x)  # [batch, seq_len, hidden_size]
        # Beta projection with sigmoid activation
        beta = torch.sigmoid(self.beta_proj(x))  # [batch, seq_len, num_heads]
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Process each head
        head_outputs = []
        for h in range(self.num_heads):
            q_h = q[:, :, h]  # [batch, seq_len, head_dim]
            k_h = k[:, :, h]  # [batch, seq_len, head_dim]
            v_h = v[:, :, h]  # [batch, seq_len, head_dim]
            beta_h = beta[:, :, h:h+1]  # [batch, seq_len, 1]
            # Apply delta rule
            out_h = self.delta_rule(q_h, k_h, v_h, beta_h)
            head_outputs.append(out_h)
        # Concatenate heads
        output = torch.cat(head_outputs, dim=-1)
        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)
        # Residual connection and layer norm
        output = self.layer_norm(residual + output)
        return output


class DeltaNetBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_hidden_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if ffn_hidden_size is None:
            ffn_hidden_size = 4 * hidden_size
        self.attention = DeltaNetLayer(hidden_size, num_heads, dropout)
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
        # Self-attention
        x = self.attention(x, mask)
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.ffn_layer_norm(residual + x)
        return x


class DeltaNetModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6, num_heads: int = 8, max_seq_len: int = 2048, dropout: float = 0.1):
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
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
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
        return f"""
DeltaNet Architecture Summary:
- Model Type: Linear Attention Transformer
- Hidden Size: {self.hidden_size}
- Number of Layers: {len(self.layers)}
- Number of Heads: {self.layers[0].attention.num_heads}
- Max Sequence Length: {self.max_seq_len}
- Key Innovation: Delta rule with forgetting mechanism
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
    # Test the model
    model = create_model(vocab_size=1000, hidden_size=256, num_layers=4)
    # Create sample input
    batch_size, seq_len = 2, 100
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(model.get_architecture_summary())
