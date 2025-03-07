"""Basic implementation of self-attention model."""

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64  # how many independent sequences to process in parallel
block_size = 256  # what is the maximum context length for predictions
max_new_tokens = 300  # how many tokens to generate when sampling from the model
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 384
num_heads = 6
n_layers = 6
dropout = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    """One self-attention head"""

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Perform the forward pass of the self-attention mechanism.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where B is the batch size,
                            T is the sequence length, and C is the number of channels.
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C) after applying self-attention.
        """

        B, T, C = x.shape
        x = torch.randn(B, T, C)

        # Single self-attention head
        k = self.key(x)
        q = self.query(x)

        # Determine the affinities between the tokens
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, 16) @ (B, 16, T) -> (B, T, T)

        # Aggregation is a data-dependent average of the keys and queries
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple parallel attention heads."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.tensor"):
        stack = torch.cat([h(x) for h in self.heads], dim=-1)
        proj = self.dropout(self.proj(stack))
        return proj


class FeedForward(nn.Module):

    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: "torch.tensor"):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embed: int, num_heads: int):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: "torch.tensor"):
        x = x + self.sa(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x


class SelfAttentionModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads=num_heads) for _ in range(n_layers)])
        self.layernorm_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: "torch.tensor", targets=None):
        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = token_embedding + position_embedding  # skip connection
        x = self.blocks(x)  # (B, T, C)
        x = self.layernorm_final(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: "torch.tensor", max_new_tokens: int):
        for _ in range(max_new_tokens):
            # crop to last block of tokens
            idx_cond = idx[:, -block_size:]
            # predictions
            logits, loss = self(idx_cond)
            # we only want the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to running sequence to get auto-regressive decoding
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
