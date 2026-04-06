"""We now want to replicate the proper transformer architecture, as detailed in paper
Attention Is All You Need. This model adds Feed Forward, Layer Normalization and
Dropout, as well as the multi-head attention from before"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.attention import MultiHeadAttention

device = "cuda" if torch.cuda.is_available() else "cpu"


class FeedForward(nn.Module):
    """
    As stated in the paper, the feed-forward network consists of two linear
    transformations with a ReLU activation in between. The ReLU function introduces
    non-linearity into networks by setting negative inputs to zero, and keeping positive
    values unchanges. A dropout is also applied to the output of each sub-layer.
    """

    def __init__(
        self,
        n_embd,
        dropout,
        d_factor=4,
    ):
        super().__init__()

        # in the paper, the inner-layer has a separate dimensionality. the paper scales
        # the model dimension up by a factor of 4. therefore we set the default factor
        # of d_factor to 4 here.
        d_ff = n_embd * d_factor

        self.network = nn.Sequential(
            nn.Linear(n_embd, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, n_embd),
            nn.Dropout(
                dropout
            ),  # dropout has a default value P_drop = 0.1 from the paper (section 5.4)
        )

    def forward(self, inputs):
        return self.network(inputs)


class TransformerBlock(nn.Module):
    """
    From the paper, we can see transformer blocks that are repeated N times. This class
    implements a single transformer block, which consists of a MultiHeadAttention block,
    LayerNorm block, Feed Forward block and another LayerNorm block
    """

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        block_size: int,
        dropout: float,
        encoder: bool = False,
    ):
        super().__init__()

        self.attention_heads = MultiHeadAttention(
            num_heads, n_embd, block_size, dropout, encoder
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)  # this LayerNorm follows attention
        self.layer_norm2 = nn.LayerNorm(n_embd)  # this LayerNorm follows feed forward

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        # in the original paper, we can see that the layernorm is applied AFTER
        # attention/feedforward components. this implementation has changed since then,
        # as it has been found that applying the layernorm before means we can train
        # models fast, and they become easier to optimize using larger learning rates
        att_out = inputs + self.layer_norm1(self.attention_heads(inputs))
        ff_out = att_out + self.layer_norm2(self.feed_forward(att_out))

        return ff_out


class TransformerLanguageModel(nn.Module):
    """
    Full transformer architecture, as detailed in the original paper. Shown in Figure 1.
    This is a decoder-only transformer! We are only generating text which isn't
    conditioned on anything. Notice that encoder = False by default in TransformerBlock.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        num_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.token_emb_table = nn.Embedding(vocab_size, n_embd)  # input embedding
        self.pos_emb_table = nn.Embedding(block_size, n_embd)  # positional encoding

        assert n_embd % num_heads == 0, "model dimension must be divisible by num_heads"

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(n_embd, num_heads, block_size, dropout)
                for _ in range(n_layers)
            ]
        )

        self.out_proj = nn.Linear(
            n_embd, vocab_size
        )  # projection back up to vocab_size to produce logits

    def forward(
        self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, block_size = inputs.shape

        tok_emb = self.token_emb_table(inputs)  # (batch_size, block_size, n_embd)
        pos_emb = self.pos_emb_table(
            torch.arange(block_size, device=device)
        )  # (block_size, n_embd)

        # combine token and positional information by broadcasting addition
        emb = tok_emb + pos_emb
        emb = self.transformer_blocks(emb)  #  apply all transformer blocks

        # project back up to vocab_size
        logits = self.out_proj(emb)  # (batch_size, block_size, vocab_size)

        loss = None
        if targets is not None:
            # same logic as in bigram model
            assert inputs.shape == targets.shape, "inputs/targets shape mismatch"
            batch_size, block_size, vocab_size = logits.shape

            loss = F.cross_entropy(
                logits.view(batch_size * block_size, vocab_size),
                targets.view(batch_size * block_size),
            )

        return logits, loss

    def generate(self, inputs: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Generate num_tokens new tokens for each independent sequence (batch)"""
        batch_size, block_size = inputs.shape

        inputs = inputs[:, -block_size:]
        output = inputs

        for _ in range(num_tokens):
            logits, _ = self(output)

            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)  # convert logits to probabilities

            # sample one new token per batch
            next_tokens = torch.multinomial(probabilities, num_samples=1)
            output = torch.cat((output, next_tokens), dim=1)  # append new token

        assert output.shape == (batch_size, block_size + num_tokens)
        return output
