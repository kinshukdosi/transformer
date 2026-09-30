"""
Tests for sampling training batches and choosing evaluation windows
"""

import torch
from data_loader import eval_windows

MAX_SEQ_LEN = 8


def test_eval_windows_cover_split():
    """With no limit, the windows should cover the tokens once, without overlapping"""

    tokens = torch.arange(100)
    inputs, targets = eval_windows(tokens, MAX_SEQ_LEN, max_windows=1000)

    # the last window needs one extra token for its target, so 99 // 8 windows fit
    assert inputs.shape == (99 // MAX_SEQ_LEN, MAX_SEQ_LEN)
    assert torch.equal(inputs.flatten().cpu(), torch.arange(len(inputs.flatten())))
    assert torch.equal(targets.cpu(), inputs.cpu() + 1)


def test_eval_windows_limit():
    """Limited windows should be non-overlapping and spread across the whole split"""

    tokens = torch.arange(10_000)
    inputs, targets = eval_windows(tokens, MAX_SEQ_LEN, max_windows=50)
    starts = inputs[:, 0].cpu()

    assert inputs.shape == (50, MAX_SEQ_LEN)
    assert torch.equal(targets.cpu(), inputs.cpu() + 1)
    assert torch.all(starts % MAX_SEQ_LEN == 0)  # aligned to non-overlapping windows
    assert torch.all(starts[1:] - starts[:-1] >= MAX_SEQ_LEN)
    assert starts[0] == 0
    # the last window that fits, with one extra token for its target
    assert starts[-1] == (len(tokens) - MAX_SEQ_LEN - 1) // MAX_SEQ_LEN * MAX_SEQ_LEN


def test_eval_windows_are_fixed():
    """Choosing evaluation windows shouldn't use the random number generator"""

    tokens = torch.arange(10_000)
    rng_state = torch.get_rng_state()
    windows_a = eval_windows(tokens, MAX_SEQ_LEN, max_windows=50)
    windows_b = eval_windows(tokens, MAX_SEQ_LEN, max_windows=50)

    assert torch.equal(torch.get_rng_state(), rng_state)
    assert torch.equal(windows_a[0], windows_b[0])
