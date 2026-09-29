"""
Tests for the training loop, checkpointing, seeding and generation scripts
"""

import torch
import train
import generate
from pathlib import Path
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
    """Training should be able to continue from a saved checkpoint"""

    cfg = get_tiny_config()
    path = tmp_path / "checkpoint.pt"
    model = train.main(cfg, output_ckpt=path)
    resumed = train.main(cfg, input_ckpt=path)

    assert_state_dicts_not_equal(model.state_dict(), resumed.state_dict())


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
