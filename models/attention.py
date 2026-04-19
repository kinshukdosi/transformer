import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

device = "cuda" if torch.cuda.is_available() else "cpu"


class Attention(nn.Module):
    """
    Forms of self-attention:

    Multi-Head Attention (set group_size to 1, num_heads == num_kv_heads)
    Multi-Query Attention (set group_size to num_heads, num_kv_heads = 1)
    Grouped Query Attention (set group_size > 1, num_kv_heads = num_heads // group_size)

    GQA divides query heads into groups, and each group shares a KV head
    This speeds up inference massively compared to MHA

    Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models
    from Multi-Head Checkpoints", 2023. https://arxiv.org/abs/2305.13245
    """

    def __init__(
        self,
        num_heads: int,
        n_embd: int,
        max_seq_len: int,
        dropout: float,
        encoder: bool = False,
        group_size: int = 1,
    ) -> None:
        super().__init__()

        assert n_embd % num_heads == 0
        assert num_heads % group_size == 0

        num_kv_heads = num_heads // group_size
        self.head_size = n_embd // num_heads
        self.group_size = group_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.query = nn.Linear(n_embd, num_heads * self.head_size, bias=False)
        self.key = nn.Linear(n_embd, num_kv_heads * self.head_size, bias=False)
        self.value = nn.Linear(n_embd, num_kv_heads * self.head_size, bias=False)
        self.output = nn.Linear(num_heads * self.head_size, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)

        # we do this to tell pytorch this tensor is part of the model but is not a
        # learnable parameter. it gets moved with the model.
        self.tril: torch.Tensor
        self.register_buffer(
            "tril", torch.tril(torch.ones(max_seq_len, max_seq_len)), persistent=False
        )

        # in "encoder" blocks of self-attention, we want all tokens to be able to attend
        # to each other, so we don't apply a mask with a triangular matrix
        self.encoder = encoder

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = inputs.shape

        q = self.query(inputs)  # (batch_size, seq_len, num_heads * head_size)
        k = self.key(inputs)  # (batch_size, seq_len, num_kv_heads * head_size)
        v = self.value(inputs)  # (batch_size, seq_len, num_kv_heads * head_size)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_size)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(
            1, 2
        )  # (batch_size, num_kv_heads, seq_len, head_size)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(
            1, 2
        )  # (batch_size, num_kv_heads, seq_len, head_size)

        # repeat KV heads to match query heads
        k = self._repeat_kv(k, self.group_size)
        v = self._repeat_kv(v, self.group_size)

        scale = 1 / self.head_size**0.5
        attn = (
            q @ k.transpose(-2, -1) * scale
        )  # (batch_size, num_heads, seq_len, seq_len)

        if not self.encoder:
            attn = attn.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (batch_size, num_heads, seq_len, head_size)

        out = (
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )  # (batch_size, seq_len, num_heads * head_size)

        return self.output(out)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        # in: (batch_size, num_kv_heads, seq_len , head_dim)
        # out: (batch_size, num_kv_heads * num_repeats, seq_len, head_dim)
        if num_repeats == 1:
            return x
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2).expand(
            batch_size, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x.reshape(batch_size, num_kv_heads * num_repeats, seq_len, head_dim)


class AttentionHeadLanguageModel(nn.Module):
    """Simple language model with only self-attention heads"""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        max_seq_len: int,
        num_heads: int,
        dropout: float,
        group_size: int = 1,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.token_emb_table = nn.Embedding(vocab_size, n_embd)
        self.pos_emb_table = nn.Embedding(
            max_seq_len, n_embd
        )  # encodes information about the position of tokens

        assert n_embd % num_heads == 0, "model dimension must be divisible by num_heads"

        self.attention = Attention(
            num_heads=num_heads,
            n_embd=n_embd,
            max_seq_len=max_seq_len,
            dropout=dropout,
            group_size=group_size,
        )  # self-attention

        self.out_proj = nn.Linear(
            n_embd, vocab_size
        )  # projection back up to vocab_size to produce logits

    def forward(
        self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seq_len = inputs.shape

        tok_emb = self.token_emb_table(inputs)  # (batch_size, seq_len, n_embd)
        pos_emb = self.pos_emb_table(
            torch.arange(seq_len, device=device)
        )  # (seq_len, n_embd)

        # combine token and positional information by broadcasting addition
        emb = tok_emb + pos_emb
        emb = self.attention(emb)  # apply self attention

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

    def generate(self, inputs: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Generate num_tokens new tokens for each independent sequence (batch)"""
        batch_size, seq_len = inputs.shape
        output = inputs

        for _ in range(num_tokens):

            # we need this here now because we now have positional embeddings, we can
            # never have more than max_seq_len tokens as an input. otherwise the table
            # would run out of scope
            sliced = output[:, -self.max_seq_len :]
            logits, _ = self(sliced)

            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)  # convert logits to probabilities

            # sample one new token per batch
            next_tokens = torch.multinomial(probabilities, num_samples=1)
            output = torch.cat((output, next_tokens), dim=1)  # append new token

        assert output.shape == (batch_size, seq_len + num_tokens)
        return output
