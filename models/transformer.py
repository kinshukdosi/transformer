"""We now want to replicate the proper transformer architecture, as detailed in paper
Attention Is All You Need. This model adds Feed Forward, Layer Normalization and
Dropout as well"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.attention import Attention

# standard deviation of the initial weights, from GPT-2. see _init_weights below
INIT_STD = 0.02


class FeedForward(nn.Module):
    """
    As stated in the paper, the feed-forward network consists of two linear
    transformations with a ReLU activation in between. The ReLU function introduces
    non-linearity into networks by setting negative inputs to zero, and keeping positive
    values unchanged. A dropout is also applied to the output of each sub-layer.
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

        # named like the projections in Attention, so the output projection of both
        # sub-layers can be found when initializing the weights
        self.hidden = nn.Linear(n_embd, d_ff)
        self.output = nn.Linear(d_ff, n_embd)

        # dropout has a default value P_drop = 0.1 from the paper (section 5.4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden(inputs))  # (batch_size, seq_len, d_ff)
        return self.dropout(self.output(hidden))  # (batch_size, seq_len, n_embd)


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
        max_seq_len: int,
        dropout: float,
        encoder: bool = False,
        group_size: int = 1,
    ):
        super().__init__()

        self.attention = Attention(
            num_heads=num_heads,
            n_embd=n_embd,
            max_seq_len=max_seq_len,
            dropout=dropout,
            encoder=encoder,
            group_size=group_size,
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)  # this LayerNorm precedes attention
        self.layer_norm2 = nn.LayerNorm(n_embd)  # this LayerNorm precedes feed forward

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        # in the original paper, we can see that the layernorm is applied AFTER
        # attention/feedforward components. this implementation has changed since then,
        # as it has been found that applying the layernorm before means we can train
        # models faster, and they become easier to optimize using larger learning rates
        att_out = inputs + self.attention(self.layer_norm1(inputs))
        ff_out = att_out + self.feed_forward(self.layer_norm2(att_out))

        return ff_out


class TransformerLanguageModel(nn.Module):
    """
    Full transformer architecture, based on the original paper. Shown in Figure 1.
    This is a decoder-only transformer! We are only generating text which isn't
    conditioned on anything. Notice that encoder = False by default in TransformerBlock.

    Like GPT-2 rather than the original paper, the blocks are Pre-LN with a final
    LayerNorm, dropout is applied to the embeddings and to the output of both
    sub-layers, and the weights are initialized with a small standard deviation.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        max_seq_len: int,
        num_heads: int,
        n_layers: int,
        dropout: float,
        group_size: int,
    ):
        super().__init__()

        self.token_emb_table = nn.Embedding(vocab_size, n_embd)  # input embedding
        self.pos_emb_table = nn.Embedding(max_seq_len, n_embd)  # positional encoding
        self.max_seq_len = max_seq_len

        assert n_embd % num_heads == 0, "model dimension must be divisible by num_heads"

        blocks = [
            TransformerBlock(
                n_embd=n_embd,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
                group_size=group_size,
            )
            for _ in range(n_layers)
        ]
        self.transformer_blocks = nn.Sequential(*blocks)

        # with Pre-LN, the residual stream itself is never normalized inside the blocks,
        # and its scale grows with every residual addition. so it is normalized once
        # more before the output projection
        self.final_layer_norm = nn.LayerNorm(n_embd)

        self.out_proj = nn.Linear(
            n_embd, vocab_size
        )  # projection back up to vocab_size to produce logits

        # dropout on the embeddings, before the first block
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

        # every block adds the output of two sub-layers to the residual stream, so its
        # variance grows with the number of layers. GPT-2 scales down the initial
        # weights of the projections that write into the residual stream by
        # 1/sqrt(number of additions), so the scale at the end doesn't depend on depth.
        # loops over the list rather than the nn.Sequential, which only knows its
        # contents are nn.Modules, so the type checker doesn't know they have .attention
        for block in blocks:
            for proj in [block.attention.output, block.feed_forward.output]:
                nn.init.normal_(proj.weight, std=INIT_STD / math.sqrt(2 * n_layers))

    @staticmethod
    def _init_weights(module: nn.Module):
        # PyTorch's defaults initialize embeddings from N(0, 1), so the embeddings
        # would start out ~50x larger than the outputs of the linear layers added to
        # them. GPT-2 initializes everything from N(0, 0.02) with zero biases instead.
        # LayerNorm keeps its default of weight 1 and bias 0
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=INIT_STD)

    def forward(
        self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seq_len = inputs.shape

        tok_emb = self.token_emb_table(inputs)  # (batch_size, seq_len, n_embd)
        pos_emb = self.pos_emb_table(
            torch.arange(seq_len, device=inputs.device)
        )  # (seq_len, n_embd)

        # combine token and positional information by broadcasting addition
        emb = self.dropout(tok_emb + pos_emb)
        emb = self.transformer_blocks(emb)  #  apply all transformer blocks
        emb = self.final_layer_norm(emb)

        # project back up to vocab_size
        logits = self.out_proj(emb)  # (batch_size, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # same logic as in bigram model
            assert inputs.shape == targets.shape, "inputs/targets shape mismatch"
            batch_size, seq_len, vocab_size = logits.shape

            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, vocab_size),
                targets.view(batch_size * seq_len),
            )

        return logits, loss

    def generate(
        self, inputs: torch.Tensor, num_tokens: int, greedy: bool = False
    ) -> torch.Tensor:
        """
        Generate num_tokens new tokens for each independent sequence (batch). Samples
        from the predicted distribution, or picks the most likely token if greedy
        """
        batch_size, seq_len = inputs.shape

        for _ in range(num_tokens):

            # we need this here now because we now have positional embeddings, we can
            # never have more than max_seq_len tokens as an input. otherwise the table
            # would run out of scope
            sliced = inputs[:, -self.max_seq_len :]
            logits, _ = self(sliced)

            logits = logits[:, -1, :]
            if greedy:
                # always the most likely token, so the output is deterministic and
                # doesn't use the random number generator
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # convert logits to probabilities
                probabilities = F.softmax(logits, dim=-1)

                # sample one new token per batch
                next_token = torch.multinomial(probabilities, num_samples=1)
            inputs = torch.cat((inputs, next_token), dim=1)  # append new token

        assert inputs.shape == (batch_size, seq_len + num_tokens)
        return inputs
