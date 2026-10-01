"""
Correctness tests for the attention and transformer language models
"""

import math
import torch
import pytest
from torch.nn import functional as F
from models.bigram import BigramLanguageModel
from models.attention import AttentionHeadLanguageModel
from models.transformer import FeedForward, TransformerBlock, TransformerLanguageModel

device = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 65
N_EMBD = 32
NUM_HEADS = 4
MAX_SEQ_LEN = 8
N_LAYERS = 2


def get_model(model_type: str, dropout: float = 0.0, group_size: int = 2):
    """Helper function for tests"""
    if model_type == "bigram":
        model = BigramLanguageModel(VOCAB_SIZE)
    elif model_type == "attention":
        model = AttentionHeadLanguageModel(
            VOCAB_SIZE, N_EMBD, MAX_SEQ_LEN, NUM_HEADS, dropout, group_size
        )
    else:
        model = TransformerLanguageModel(
            VOCAB_SIZE, N_EMBD, MAX_SEQ_LEN, NUM_HEADS, N_LAYERS, dropout, group_size
        )
    return model.to(device)


def test_feed_forward_dimensions():
    """Inner layer of the feed forward network should be 4 * n_embd"""

    feed_forward = FeedForward(N_EMBD, dropout=0.0)

    assert feed_forward.hidden.weight.shape == (4 * N_EMBD, N_EMBD)
    assert feed_forward.output.weight.shape == (N_EMBD, 4 * N_EMBD)
    assert feed_forward(torch.randn(2, MAX_SEQ_LEN, N_EMBD)).shape == (
        2,
        MAX_SEQ_LEN,
        N_EMBD,
    )


def test_transformer_block_pre_layer_norm():
    """
    Test normalization placement and residual connections in a transformer block. The
    block should compute:
        x = x + attention(layer_norm1(x))
        x = x + feed_forward(layer_norm2(x))
    """

    torch.manual_seed(0)
    block = TransformerBlock(
        n_embd=N_EMBD, num_heads=NUM_HEADS, max_seq_len=MAX_SEQ_LEN, dropout=0.0
    ).to(device)

    # record the input and output of each sub-layer during the forward pass
    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            captured[name] = (inputs[0], output)

        return hook

    for name in ["layer_norm1", "attention", "layer_norm2", "feed_forward"]:
        getattr(block, name).register_forward_hook(capture(name))

    inputs = torch.randn(2, MAX_SEQ_LEN, N_EMBD, device=device)
    with torch.no_grad():
        out = block(inputs)

    ln1_in, ln1_out = captured["layer_norm1"]
    att_in, att_out = captured["attention"]
    ln2_in, ln2_out = captured["layer_norm2"]
    ff_in, ff_out = captured["feed_forward"]

    assert torch.equal(ln1_in, inputs)  # layer norm is applied to the block input
    assert torch.equal(att_in, ln1_out)  # attention sees the normalized input
    assert torch.allclose(ln2_in, inputs + att_out)  # first residual connection
    assert torch.equal(ff_in, ln2_out)  # feed forward sees the normalized input
    assert torch.allclose(out, ln2_in + ff_out)  # second residual connection


def test_transformer_final_layer_norm():
    """The output of the last block is normalized before the output projection"""

    torch.manual_seed(0)
    model = get_model("transformer").eval()

    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            captured[name] = (inputs[0], output)

        return hook

    for name in ["transformer_blocks", "final_layer_norm", "out_proj"]:
        getattr(model, name).register_forward_hook(capture(name))

    inputs = torch.randint(VOCAB_SIZE, (2, MAX_SEQ_LEN), device=device)
    with torch.no_grad():
        model(inputs)

    _, blocks_out = captured["transformer_blocks"]
    ln_in, ln_out = captured["final_layer_norm"]
    proj_in, _ = captured["out_proj"]

    assert torch.equal(ln_in, blocks_out)
    assert torch.equal(proj_in, ln_out)


def test_transformer_init():
    """
    Weights start from N(0, 0.02) with zero biases, except the projections into the
    residual stream, which are scaled down by sqrt(2 * n_layers)
    """

    torch.manual_seed(0)
    model = TransformerLanguageModel(
        VOCAB_SIZE,
        n_embd=64,
        max_seq_len=MAX_SEQ_LEN,
        num_heads=NUM_HEADS,
        n_layers=N_LAYERS,
        dropout=0.0,
        group_size=1,
    )
    residual_std = 0.02 / math.sqrt(2 * N_LAYERS)

    def assert_std(weight: torch.Tensor, std: float):
        assert weight.mean().abs() < std / 5
        assert weight.std().item() == pytest.approx(std, rel=0.1)

    assert_std(model.token_emb_table.weight, 0.02)
    assert_std(model.pos_emb_table.weight, 0.02)
    assert_std(model.out_proj.weight, 0.02)
    assert torch.equal(model.out_proj.bias, torch.zeros_like(model.out_proj.bias))

    for block in model.transformer_blocks:
        assert isinstance(block, TransformerBlock)
        assert_std(block.attention.query.weight, 0.02)
        assert_std(block.feed_forward.hidden.weight, 0.02)
        assert_std(block.attention.output.weight, residual_std)
        assert_std(block.feed_forward.output.weight, residual_std)
        for linear in [block.feed_forward.hidden, block.feed_forward.output]:
            assert torch.equal(linear.bias, torch.zeros_like(linear.bias))
        assert torch.equal(block.layer_norm1.weight, torch.ones(64))


def test_attention_output_dropout():
    """Dropout is applied to the attention output, after the output projection"""

    torch.manual_seed(0)
    block = TransformerBlock(
        n_embd=N_EMBD, num_heads=NUM_HEADS, max_seq_len=MAX_SEQ_LEN, dropout=0.5
    ).to(device)

    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output

        return hook

    block.attention.output.register_forward_hook(capture("projection"))
    block.attention.register_forward_hook(capture("attention"))

    with torch.no_grad():
        block(torch.randn(2, MAX_SEQ_LEN, N_EMBD, device=device))

    projection, attention = captured["projection"], captured["attention"]
    dropped = attention == 0

    # roughly half the outputs are dropped, the rest are scaled by 1 / (1 - p)
    assert 0.3 < dropped.float().mean() < 0.7
    assert torch.allclose(attention[~dropped], 2 * projection[~dropped])


def test_embedding_dropout():
    """Dropout is applied to the embeddings in training, and not in evaluation"""

    torch.manual_seed(0)
    model = get_model("transformer", dropout=0.5)
    assert isinstance(model, TransformerLanguageModel)

    captured = []
    model.transformer_blocks.register_forward_hook(
        lambda module, inputs, output: captured.append(inputs[0])
    )

    inputs = torch.randint(VOCAB_SIZE, (2, MAX_SEQ_LEN), device=device)
    with torch.no_grad():
        model.train()
        model(inputs)
        model.eval()
        model(inputs)

    train_emb, eval_emb = captured
    dropped = train_emb == 0

    assert 0.3 < dropped.float().mean() < 0.7
    assert torch.allclose(train_emb[~dropped], 2 * eval_emb[~dropped])
    assert not (eval_emb == 0).any()


@pytest.mark.parametrize(("model_type"), ["bigram", "attention", "transformer"])
def test_model_loss(model_type):
    """Test logits shape and that loss is the mean cross entropy over all positions"""

    torch.manual_seed(0)
    model = get_model(model_type)
    seq_len = 1 if model_type == "bigram" else MAX_SEQ_LEN
    inputs = torch.randint(VOCAB_SIZE, (4, seq_len), device=device)
    targets = torch.randint(VOCAB_SIZE, (4, seq_len), device=device)

    with torch.no_grad():
        logits, loss = model(inputs, targets)
        _, no_loss = model(inputs)

    log_probs = F.log_softmax(logits, dim=-1)
    expected = -log_probs.gather(-1, targets.unsqueeze(-1)).mean()

    assert logits.shape == (4, seq_len, VOCAB_SIZE)
    assert no_loss is None
    assert torch.allclose(loss, expected, atol=1e-5)

    # an untrained model should be close to a uniform distribution over the vocab
    assert abs(loss.item() - math.log(VOCAB_SIZE)) < 1


@pytest.mark.parametrize(("model_type"), ["attention", "transformer"])
def test_model_is_causal(model_type):
    """Changing a token shouldn't change the logits of any earlier position"""

    torch.manual_seed(0)
    model = get_model(model_type).eval()
    inputs = torch.randint(VOCAB_SIZE, (2, MAX_SEQ_LEN), device=device)
    position = 5

    changed = inputs.clone()
    changed[:, position] = (changed[:, position] + 1) % VOCAB_SIZE

    with torch.no_grad():
        logits, _ = model(inputs)
        logits_changed, _ = model(changed)

    assert torch.equal(logits[:, :position], logits_changed[:, :position])
    assert not torch.allclose(logits[:, position], logits_changed[:, position])


@pytest.mark.parametrize(("group_size"), [1, 2, 4])
def test_transformer_gradients(group_size):
    """Every parameter should receive a finite, non-zero gradient"""

    torch.manual_seed(0)
    model = get_model("transformer", group_size=group_size)
    inputs = torch.randint(VOCAB_SIZE, (4, MAX_SEQ_LEN), device=device)
    targets = torch.randint(VOCAB_SIZE, (4, MAX_SEQ_LEN), device=device)

    _, loss = model(inputs, targets)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
        assert param.grad.abs().sum() > 0, f"{name} has zero gradient"


def test_transformer_overfits_single_batch():
    """A working model and training step should be able to memorise one batch"""

    torch.manual_seed(0)
    model = get_model("transformer")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inputs = torch.randint(VOCAB_SIZE, (4, MAX_SEQ_LEN), device=device)
    targets = torch.randint(VOCAB_SIZE, (4, MAX_SEQ_LEN), device=device)

    for _ in range(200):
        _, loss = model(inputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # measure the loss after the final optimizer step
    with torch.no_grad():
        _, loss = model(inputs, targets)

    assert loss.item() < 0.1


@pytest.mark.parametrize(("model_type"), ["attention", "transformer"])
def test_model_on_cpu(model_type):
    """Models should run on the device of their inputs, even when CUDA is available"""

    model = get_model(model_type).cpu()
    inputs = torch.randint(VOCAB_SIZE, (2, MAX_SEQ_LEN))
    logits, _ = model(inputs)

    assert logits.device.type == "cpu"


@pytest.mark.parametrize(("model_type"), ["bigram", "attention", "transformer"])
def test_generate(model_type):
    """
    Test generation keeps the prompt, produces valid token ids, and can generate past
    max_seq_len
    """

    torch.manual_seed(0)
    model = get_model(model_type).eval()
    prompt = torch.randint(VOCAB_SIZE, (2, 3), device=device)
    num_tokens = 3 * MAX_SEQ_LEN

    with torch.no_grad():
        output = model.generate(prompt, num_tokens=num_tokens)

    assert output.shape == (2, 3 + num_tokens)
    assert torch.equal(output[:, :3], prompt)
    assert output.min() >= 0 and output.max() < VOCAB_SIZE


def test_generate_is_seeded():
    """Sampling with the same seed should give the same output"""

    model = get_model("transformer").eval()
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)

    outputs = []
    for _ in range(2):
        torch.manual_seed(0)
        with torch.no_grad():
            outputs.append(model.generate(prompt, num_tokens=20))

    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize(("model_type"), ["bigram", "attention", "transformer"])
def test_generate_greedy(model_type):
    """
    Greedy generation picks the most likely token at every step, gives the same output
    whatever the seed, and doesn't use the random number generator
    """

    model = get_model(model_type).eval()
    prompt = torch.randint(VOCAB_SIZE, (2, 3), device=device)
    num_tokens = 2 * MAX_SEQ_LEN

    outputs = []
    for seed in [0, 1]:
        torch.manual_seed(seed)
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all()
        with torch.no_grad():
            outputs.append(model.generate(prompt, num_tokens=num_tokens, greedy=True))
        assert torch.equal(torch.get_rng_state(), rng_state)
        assert all(
            torch.equal(a, b)
            for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng_state)
        )

    output = outputs[0]
    assert torch.equal(output, outputs[1])

    # every new token is the argmax of the prediction for the context before it
    for i in range(3, 3 + num_tokens):
        context = output[:, :i][:, -MAX_SEQ_LEN:]
        with torch.no_grad():
            logits, _ = model(context)
        assert torch.equal(output[:, i], logits[:, -1, :].argmax(dim=-1))
