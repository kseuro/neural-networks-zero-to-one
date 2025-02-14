import torch
import torch.nn as nn
from torch.nn import functional as F

n_embed = 32
block_size = 8


class Head(nn.Module):
    """One self-attention head"""

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

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

        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
