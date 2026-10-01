"""
Tests for the training loop, checkpointing, seeding and generation scripts
"""

import math
import hashlib
import torch
import train
import pytest
import generate
from pathlib import Path
from results import get_commit_hash, load_runs
from config import (
    device,
    DEFAULT_SEED,
    DEFAULT_OPTIMIZER_SETTINGS,
    DEFAULT_SCHEDULE_SETTINGS,
    BaseConfig,
    parse_config,
    get_config_from_pytorch_model,
    get_model_from_config,
    get_optimizer_from_config,
)

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR.parent / "data"


def get_tiny_config(**overrides) -> BaseConfig:
    """Helper function for tests. A tiny transformer that trains in seconds"""
    cfg_dict = {
        "tokenizer": "simple",
        "model_type": "transformer",
        "vocab_size": 65,
        "data_path": str(DATA_DIR / "tinyshakespeare.txt"),
        "max_seq_len": 8,
        "train_split": 0.9,
        "batch_size": 4,
        "iterations": 20,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "eval_iterations": 2,
        "eval_interval": 10,
        "seed": 1337,
        "n_embd": 32,
        "num_heads": 4,
        "dropout": 0.1,
        "group_size": 2,
        "n_layers": 2,
    }
    cfg_dict.update(overrides)
    return parse_config(cfg_dict)


def assert_state_dicts_equal(a: dict, b: dict):
    """Helper function for tests"""
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} differs"


def assert_state_dicts_not_equal(a: dict, b: dict):
    """Helper function for tests"""
    assert a.keys() == b.keys()
    for key in a:
        assert not torch.equal(a[key], b[key]), f"{key} is unchanged"


def test_seed_default():
    """Configs and checkpoints saved before seed was configurable use the old seed"""

    cfg = get_tiny_config()
    cfg_dict = vars(cfg).copy()
    del cfg_dict["seed"]

    assert parse_config(cfg_dict).seed == DEFAULT_SEED


def test_optimizer_defaults():
    """
    Configs and checkpoints saved before the optimizer settings were configurable use
    PyTorch's AdamW defaults, so they train exactly as before
    """

    cfg_dict = vars(get_tiny_config()).copy()
    for key in DEFAULT_OPTIMIZER_SETTINGS:
        del cfg_dict[key]
    cfg = parse_config(cfg_dict)
    defaults = torch.optim.AdamW([torch.zeros(1)]).defaults

    assert (cfg.beta1, cfg.beta2) == defaults["betas"]
    assert cfg.eps == defaults["eps"]
    assert cfg.weight_decay == defaults["weight_decay"]


def test_optimizer_settings(tmp_path):
    """Optimizer settings in the config should be the ones used for training"""

    cfg = get_tiny_config(beta1=0.8, beta2=0.95, eps=1e-6, weight_decay=0.1)
    path = tmp_path / "checkpoint.pt"
    train.main(cfg, output_ckpt=path)
    decay_group, no_decay_group = torch.load(path)["optimizer_state_dict"][
        "param_groups"
    ]

    for param_group in [decay_group, no_decay_group]:
        assert param_group["betas"] == (0.8, 0.95)
        assert param_group["eps"] == 1e-6
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0


def test_schedule_defaults():
    """
    Without schedule settings in the config, the learning rate is constant and
    gradients aren't clipped, which is how runs trained before they existed
    """

    cfg_dict = vars(get_tiny_config()).copy()
    for key in DEFAULT_SCHEDULE_SETTINGS:
        del cfg_dict[key]
    cfg = parse_config(cfg_dict)

    assert cfg.grad_clip is None
    assert [train.get_lr(cfg, step) for step in [0, 10, 1000]] == [cfg.lr] * 3


def test_get_lr():
    """Linear warmup, then cosine decay to min_lr, then constant at min_lr"""

    cfg = get_tiny_config(lr=1e-3, warmup_iters=10, lr_decay_iters=110, min_lr=1e-4)

    # warmup: the first step already has a non-zero learning rate
    assert train.get_lr(cfg, 0) == pytest.approx(1e-4)
    assert train.get_lr(cfg, 4) == pytest.approx(5e-4)
    assert train.get_lr(cfg, 9) == pytest.approx(1e-3)

    # cosine decay: lr at the start, halfway between lr and min_lr in the middle
    assert train.get_lr(cfg, 10) == pytest.approx(1e-3)
    assert train.get_lr(cfg, 60) == pytest.approx(5.5e-4)
    assert train.get_lr(cfg, 110) == pytest.approx(1e-4)
    assert train.get_lr(cfg, 1000) == pytest.approx(1e-4)

    lrs = [train.get_lr(cfg, step) for step in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))  # never increases while decaying

    # warmup only
    cfg = get_tiny_config(lr=1e-3, warmup_iters=10)
    assert train.get_lr(cfg, 1000) == pytest.approx(1e-3)


def test_lr_used_for_training(tmp_path):
    """The optimizer uses the scheduled learning rate, and it is recorded in results"""

    cfg = get_tiny_config(warmup_iters=5, lr_decay_iters=15, min_lr=1e-4)
    path = tmp_path / "checkpoint.pt"
    train.main(cfg, output_ckpt=path, results_dir=tmp_path)

    # the last step trained was iterations - 1, after the decay finished
    for param_group in torch.load(path)["optimizer_state_dict"]["param_groups"]:
        assert param_group["lr"] == pytest.approx(1e-4)

    [run] = load_runs(tmp_path)
    for e in run["evals"]:
        assert e["lr"] == pytest.approx(train.get_lr(cfg, e["step"]))


@pytest.mark.parametrize(
    ("overrides"),
    [
        {"warmup_iters": -1},
        {"lr_decay_iters": 100},  # without min_lr
        {"min_lr": 1e-4},  # without lr_decay_iters
        {"warmup_iters": 100, "lr_decay_iters": 50, "min_lr": 1e-4},
        {"lr_decay_iters": 100, "min_lr": 1e-2},  # above lr
        {"grad_clip": 0.0},
    ],
)
def test_invalid_schedule(overrides):
    with pytest.raises(AssertionError):
        train.main(get_tiny_config(**overrides))


def test_weight_decay_groups():
    """Weight matrices and embeddings are decayed, biases and LayerNorm aren't"""

    cfg = get_tiny_config(weight_decay=0.1)
    model = get_model_from_config(cfg)
    decay_group, no_decay_group = get_optimizer_from_config(cfg, model).param_groups

    names = {param: name for name, param in model.named_parameters()}
    decayed = {names[p] for p in decay_group["params"]}
    not_decayed = {names[p] for p in no_decay_group["params"]}

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert decayed | not_decayed == set(names.values())  # every parameter, once
    assert decayed & not_decayed == set()

    assert "token_emb_table.weight" in decayed
    assert "out_proj.weight" in decayed
    assert "transformer_blocks.0.attention.query.weight" in decayed
    assert "out_proj.bias" in not_decayed
    assert "final_layer_norm.weight" in not_decayed
    assert "transformer_blocks.0.layer_norm1.bias" in not_decayed


def test_grad_clip():
    """With grad_clip, the combined norm of the gradients used for the update is
    at most grad_clip"""

    inputs = torch.randint(65, (4, 8), device=device)
    targets = torch.randint(65, (4, 8), device=device)

    def grad_norm(grad_clip):
        torch.manual_seed(0)
        cfg = get_tiny_config(grad_clip=grad_clip)
        model = get_model_from_config(cfg)
        train.training_step(
            model, get_optimizer_from_config(cfg, model), cfg, inputs, targets
        )
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        return torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item()

    unclipped = grad_norm(None)
    assert unclipped > 0.1
    assert grad_norm(0.1) == pytest.approx(0.1, rel=1e-3)
    assert grad_norm(1e6) == pytest.approx(unclipped)  # below grad_clip, unchanged


def test_training_is_seeded():
    """Two runs with the same seed should give identical weights"""

    model_a = train.main(get_tiny_config())
    model_b = train.main(get_tiny_config())
    model_c = train.main(get_tiny_config(seed=2024))

    assert_state_dicts_equal(model_a.state_dict(), model_b.state_dict())
    assert_state_dicts_not_equal(model_a.state_dict(), model_c.state_dict())


def test_evaluation_does_not_change_training(tmp_path):
    """
    Evaluating more often, or on more batches, shouldn't change the trained model, and
    the losses at steps both runs evaluated should be identical
    """

    model_a = train.main(get_tiny_config(), results_dir=tmp_path / "a")
    model_b = train.main(get_tiny_config(eval_interval=5), results_dir=tmp_path / "b")
    model_c = train.main(get_tiny_config(eval_iterations=4))

    assert_state_dicts_equal(model_a.state_dict(), model_b.state_dict())
    assert_state_dicts_equal(model_a.state_dict(), model_c.state_dict())

    [run_a] = load_runs(tmp_path / "a")
    [run_b] = load_runs(tmp_path / "b")
    assert [e["step"] for e in run_b["evals"]] == [0, 5, 10, 15, 20]
    assert run_a["evals"] == run_b["evals"][::2]  # steps 0, 10 and 20


def test_tokenizer_not_trained_on_validation_text(tmp_path):
    """The tokenizer should only learn its vocabulary from the training split"""

    # "z" only appears in the last 10% of the text, which is the validation split
    data_path = tmp_path / "data.txt"
    data_path.write_text("abcd" * 900 + "z" * 400)

    # the character tokenizer has no unknown token, so encoding the validation text
    # fails if "z" was not in the text the tokenizer was trained on
    with pytest.raises(KeyError):
        train.main(get_tiny_config(data_path=str(data_path)))


def test_checkpoint_round_trip(tmp_path):
    """Saved checkpoint should restore the config, model and optimizer state"""

    cfg = get_tiny_config()
    path = tmp_path / "checkpoint.pt"
    model = train.main(cfg, output_ckpt=path)

    checkpoint = torch.load(path)
    restored_cfg = get_config_from_pytorch_model(path)
    restored_model = get_model_from_config(restored_cfg)
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    optim = get_optimizer_from_config(restored_cfg, restored_model)
    optim.load_state_dict(checkpoint["optimizer_state_dict"])

    assert restored_cfg == cfg
    assert_state_dicts_equal(restored_model.state_dict(), model.state_dict())

    # AdamW state for every parameter, having taken one step per training iteration
    assert len(optim.state) == len(list(restored_model.parameters()))
    for state in optim.state.values():
        assert state["step"].item() == cfg.iterations
        assert state["exp_avg"].abs().sum() > 0


@pytest.mark.parametrize(
    ("schedule"),
    [{}, {"warmup_iters": 5, "lr_decay_iters": 15, "min_lr": 1e-4, "grad_clip": 0.5}],
)
def test_resume_from_checkpoint(tmp_path, schedule):
    """
    Training for 10 steps, saving, then resuming up to 20 steps should give exactly the
    same weights and losses as training for 20 steps without stopping. With a learning
    rate schedule, the checkpoint is saved halfway through the decay
    """

    uninterrupted = train.main(
        get_tiny_config(iterations=20, **schedule),
        results_dir=tmp_path / "uninterrupted",
    )

    path = tmp_path / "checkpoint.pt"
    train.main(get_tiny_config(iterations=10, **schedule), output_ckpt=path)
    resumed = train.main(
        get_tiny_config(iterations=20, **schedule),
        input_ckpt=path,
        results_dir=tmp_path / "resumed",
    )

    [uninterrupted_run] = load_runs(tmp_path / "uninterrupted")
    [resumed_run] = load_runs(tmp_path / "resumed")

    # without this, a run that ignored the checkpoint and trained from scratch would
    # also match, because every call to main() is seeded the same way
    assert resumed_run["training"]["start_step"] == 10
    assert resumed_run["training"]["steps"] == 10

    assert_state_dicts_equal(uninterrupted.state_dict(), resumed.state_dict())
    assert resumed_run["evals"] == uninterrupted_run["evals"][1:]  # steps 10 and 20


def test_run_results(tmp_path):
    """Each run should save a results file with its metadata and measurements"""

    cfg = get_tiny_config(experiment="tiny-test")
    model = train.main(cfg, results_dir=tmp_path)
    [run] = load_runs(tmp_path)
    with open(cfg.data_path, "rb") as f:
        data = f.read()

    assert run["experiment"] == "tiny-test"
    assert "tiny-test" in run["run_id"]
    assert run["dataset"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert run["dataset"]["bytes"] == len(data)
    assert run["git_commit"] == get_commit_hash()
    assert run["config"]["seed"] == cfg.seed
    assert run["system"]["torch"] == torch.__version__
    assert run["parameters"] == sum(p.numel() for p in model.parameters())

    # evaluated every eval_interval steps, and once more after the final step
    assert [e["step"] for e in run["evals"]] == [0, 10, 20]
    assert run["final"]["step"] == cfg.iterations
    assert run["final"]["val_perplexity"] == pytest.approx(
        math.exp(run["final"]["val_loss"])
    )

    training = run["training"]
    assert training["tokens"] == cfg.iterations * cfg.batch_size * cfg.max_seq_len
    assert training["tokens_per_sec"] > 0
    if torch.cuda.is_available():
        assert training["peak_memory_mb"] > 0


def test_generate_script(tmp_path, capsys):
    """End-to-end: train, save, then generate text from a prompt"""

    path = tmp_path / "checkpoint.pt"
    train.main(get_tiny_config(), output_ckpt=path)
    capsys.readouterr()  # discard training output

    prompt = "ROMEO:"
    generate.main(path, prompt)
    output = capsys.readouterr().out

    assert output.startswith(prompt)
    assert len(output) > len(prompt) + 100


def test_generate_script_options(tmp_path, capsys):
    """The script generates num_tokens tokens, and greedy output ignores the seed"""

    path = tmp_path / "checkpoint.pt"
    train.main(get_tiny_config(), output_ckpt=path)
    capsys.readouterr()  # discard training output

    prompt = "ROMEO:"
    outputs = []
    for seed in [0, 1]:
        torch.manual_seed(seed)
        generate.main(path, prompt, num_tokens=50, greedy=True)
        outputs.append(capsys.readouterr().out)

    # the tiny config uses the character tokenizer, so one token is one character.
    # print() adds the final newline
    assert len(outputs[0]) == len(prompt) + 50 + 1
    assert outputs[0] == outputs[1]


def test_checkpoint_tokenizer(tmp_path, capsys):
    """Generating from a checkpoint shouldn't need the dataset the tokenizer was
    trained on"""

    data_path = tmp_path / "data.txt"
    data_path.write_text((DATA_DIR / "tinyshakespeare.txt").read_text())
    path = tmp_path / "checkpoint.pt"
    train.main(get_tiny_config(data_path=str(data_path)), output_ckpt=path)
    data_path.unlink()
    capsys.readouterr()  # discard training output

    prompt = "ROMEO:"
    generate.main(path, prompt)

    assert capsys.readouterr().out.startswith(prompt)


def test_generate_old_checkpoint(tmp_path, capsys):
    """Checkpoints saved before the tokenizer was stored should retrain it"""

    path = tmp_path / "checkpoint.pt"
    train.main(get_tiny_config(), output_ckpt=path)
    checkpoint = torch.load(path)
    del checkpoint["tokenizer"]
    torch.save(checkpoint, path)
    capsys.readouterr()  # discard training output

    prompt = "ROMEO:"
    generate.main(path, prompt)

    assert capsys.readouterr().out.startswith(prompt)
