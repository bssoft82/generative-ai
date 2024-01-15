from engine.FeedForward import FeedForward
from engine.MultiHeadAttention import MultiHeadAttention
import torch.nn as nn


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_heads, n_embd, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
