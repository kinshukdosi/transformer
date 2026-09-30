"""
Tests for comparing run results
"""

import yaml
import train
import pytest
from pathlib import Path
from config import DEFAULT_OPTIMIZER_SETTINGS, parse_config
from results import load_runs
from compare import compare_runs, comparability_warnings, config_differences

TEST_DIR = Path(__file__).parent
CONFIG_DIR = TEST_DIR.parent / "config"


def get_runs(tmp_path: Path) -> list[dict]:
    """Helper function for tests. Two short runs that only differ in learning rate"""
    for lr in [1e-3, 3e-3]:
        with open(CONFIG_DIR / "debug" / "transformer.yaml", "r") as f:
            cfg = parse_config(yaml.safe_load(f))
        cfg.iterations = 10
        cfg.lr = lr
        train.main(cfg, results_dir=tmp_path)

    runs = load_runs(tmp_path)
    # the tests themselves usually run with uncommitted changes
    for run in runs:
        run["git_dirty"] = False
    return runs


def test_compare_runs(tmp_path):
    """Only the settings that differ get a column, and every run gets a row"""

    runs = get_runs(tmp_path)
    table = compare_runs(runs)

    assert config_differences(runs) == ["lr"]
    assert comparability_warnings(runs) == []
    for run in runs:
        assert run["run_id"] in table
    assert "Warning" not in table


def test_compare_old_runs(tmp_path):
    """Runs saved before newer settings existed are compared using their defaults"""

    runs = get_runs(tmp_path)
    del runs[0]["dataset"]
    del runs[0]["experiment"]
    for key in DEFAULT_OPTIMIZER_SETTINGS:
        del runs[0]["config"][key]

    assert config_differences(runs) == ["lr"]
    assert "dataset was recorded" in comparability_warnings(runs)[0]


def set_dataset(run: dict):
    """Helper function for tests"""
    run["dataset"]["sha256"] = "0" * 64


def set_vocab_size(run: dict):
    """Helper function for tests"""
    run["config"]["vocab_size"] += 1


def set_gpu(run: dict):
    """Helper function for tests"""
    run["system"]["gpu"] = "another GPU"


def set_dirty(run: dict):
    """Helper function for tests"""
    run["git_dirty"] = True


@pytest.mark.parametrize(
    ("change", "warning"),
    [
        (set_dataset, "different datasets"),
        (set_vocab_size, "different tokenizers"),
        (set_gpu, "different hardware"),
        (set_dirty, "uncommitted changes"),
    ],
)
def test_comparability_warnings(tmp_path, change, warning):
    """Differences that make runs not directly comparable should be warned about"""

    runs = get_runs(tmp_path)
    change(runs[1])
    warnings = comparability_warnings(runs)

    assert len(warnings) == 1
    assert warning in warnings[0]
    assert warning in compare_runs(runs)
