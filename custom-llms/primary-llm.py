import torch
import torch.nn as nn
from torch.optim import Adam


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, block_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.embed(idx)
        pos_emb = self.pos_embed(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.head(x)
        return logits