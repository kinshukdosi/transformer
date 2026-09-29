"""
Correctness tests for the Attention module (MHA, GQA and MQA)
"""

import torch
import pytest
from torch.nn import functional as F
from models.attention import Attention

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_HEADS = 4
N_EMBD = 32
MAX_SEQ_LEN = 8


def get_attention(group_size: int) -> Attention:
    """Helper function for tests"""
    attention = Attention(
        num_heads=NUM_HEADS,
        n_embd=N_EMBD,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
        group_size=group_size,
    )
    return attention.to(device).eval()


def reference_attention(attention: Attention, inputs: torch.Tensor) -> torch.Tensor:
    """
    Helper function for tests. Computes causal attention with the module's weights, but
    using PyTorch's scaled_dot_product_attention and repeat_interleave instead of the
    module's own masking and _repeat_kv logic
    """
    batch_size, seq_len, n_embd = inputs.shape
    head_size = attention.head_size

    q = attention.query(inputs).view(batch_size, seq_len, -1, head_size).transpose(1, 2)
    k = attention.key(inputs).view(batch_size, seq_len, -1, head_size).transpose(1, 2)
    v = attention.value(inputs).view(batch_size, seq_len, -1, head_size).transpose(1, 2)

    # query heads [0, 1, ..., group_size - 1] share KV head 0, and so on
    k = k.repeat_interleave(attention.group_size, dim=1)
    v = v.repeat_interleave(attention.group_size, dim=1)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).reshape(batch_size, seq_len, n_embd)
    return attention.output(out)


@pytest.mark.parametrize(("group_size"), [1, 2, 4])
def test_attention_shapes(group_size):
    """Test projection and output shapes for MHA (1), GQA (2) and MQA (4)"""

    attention = get_attention(group_size)
    num_kv_heads = NUM_HEADS // group_size
    head_size = N_EMBD // NUM_HEADS

    assert attention.num_kv_heads == num_kv_heads
    assert attention.query.weight.shape == (NUM_HEADS * head_size, N_EMBD)
    assert attention.key.weight.shape == (num_kv_heads * head_size, N_EMBD)
    assert attention.value.weight.shape == (num_kv_heads * head_size, N_EMBD)
    assert attention.output.weight.shape == (N_EMBD, NUM_HEADS * head_size)

    # seq_len can be anything up to max_seq_len
    for seq_len in [1, MAX_SEQ_LEN // 2, MAX_SEQ_LEN]:
        inputs = torch.randn(2, seq_len, N_EMBD, device=device)
        assert attention(inputs).shape == (2, seq_len, N_EMBD)


@pytest.mark.parametrize(("group_size"), [1, 2, 4])
def test_attention_matches_reference(group_size):
    """Test head splitting, KV head sharing and head merging against a reference"""

    torch.manual_seed(0)
    attention = get_attention(group_size)
    inputs = torch.randn(2, MAX_SEQ_LEN, N_EMBD, device=device)

    with torch.no_grad():
        out = attention(inputs)
        expected = reference_attention(attention, inputs)

    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize(("group_size"), [1, 2, 4])
def test_attention_is_causal(group_size):
    """Test that changing a token doesn't change the outputs of earlier positions"""

    torch.manual_seed(0)
    attention = get_attention(group_size)
    inputs = torch.randn(2, MAX_SEQ_LEN, N_EMBD, device=device)
    position = 5

    changed = inputs.clone()
    changed[:, position] = torch.randn(2, N_EMBD, device=device)

    with torch.no_grad():
        out = attention(inputs)
        out_changed = attention(changed)

    assert torch.equal(out[:, :position], out_changed[:, :position])
    assert not torch.allclose(out[:, position:], out_changed[:, position:])


@pytest.mark.parametrize(("group_size"), [2, 4])
def test_gqa_equals_mha_with_shared_kv_heads(group_size):
    """
    GQA/MQA should be exactly equivalent to MHA where every query head in a group has
    identical key and value weights
    """

    torch.manual_seed(0)
    gqa = get_attention(group_size)
    mha = get_attention(1)

    head_size = N_EMBD // NUM_HEADS
    with torch.no_grad():
        mha.query.weight.copy_(gqa.query.weight)
        mha.output.weight.copy_(gqa.output.weight)
        for name in ["key", "value"]:
            # (num_kv_heads * head_size, n_embd) -> (num_heads * head_size, n_embd)
            weight = getattr(gqa, name).weight.view(-1, head_size, N_EMBD)
            weight = weight.repeat_interleave(group_size, dim=0)
            getattr(mha, name).weight.copy_(weight.reshape(-1, N_EMBD))

        inputs = torch.randn(2, MAX_SEQ_LEN, N_EMBD, device=device)
        assert torch.allclose(gqa(inputs), mha(inputs), atol=1e-6)


def test_attention_invalid_group_size():
    """num_heads must be divisible by group_size"""

    with pytest.raises(AssertionError):
        Attention(
            num_heads=NUM_HEADS,
            n_embd=N_EMBD,
            max_seq_len=MAX_SEQ_LEN,
            dropout=0.0,
            group_size=3,
        )
