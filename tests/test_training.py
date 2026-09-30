"""
Tests for the training loop, checkpointing, seeding and generation scripts
"""

import math
import torch
import train
import pytest
import generate
from pathlib import Path
from results import get_commit_hash, load_runs
from config import (
    DEFAULT_SEED,
    BaseConfig,
    parse_config,
    get_config_from_pytorch_model,
    get_model_from_config,
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
    optim = torch.optim.AdamW(restored_model.parameters(), lr=restored_cfg.lr)
    optim.load_state_dict(checkpoint["optimizer_state_dict"])

    assert restored_cfg == cfg
    assert_state_dicts_equal(restored_model.state_dict(), model.state_dict())

    # AdamW state for every parameter, having taken one step per training iteration
    assert len(optim.state) == len(list(restored_model.parameters()))
    for state in optim.state.values():
        assert state["step"].item() == cfg.iterations
        assert state["exp_avg"].abs().sum() > 0


def test_resume_from_checkpoint(tmp_path):
    """
    Training for 10 steps, saving, then resuming up to 20 steps should give exactly the
    same weights and losses as training for 20 steps without stopping
    """

    uninterrupted = train.main(
        get_tiny_config(iterations=20), results_dir=tmp_path / "uninterrupted"
    )

    path = tmp_path / "checkpoint.pt"
    train.main(get_tiny_config(iterations=10), output_ckpt=path)
    resumed = train.main(
        get_tiny_config(iterations=20),
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

    cfg = get_tiny_config()
    model = train.main(cfg, results_dir=tmp_path)
    [run] = load_runs(tmp_path)

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
