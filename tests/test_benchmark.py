"""
Tests for the benchmark runner
"""

import yaml
import train
import pytest
import benchmark
from pathlib import Path
from config import parse_config
from results import count_parameters, get_model_memory_mb, load_runs
from models.transformer import TransformerLanguageModel

TEST_DIR = Path(__file__).parent
CONFIG_DIR = TEST_DIR.parent / "config"


def get_debug_config(model_type: str):
    """Helper function for tests"""
    with open(CONFIG_DIR / "debug" / f"{model_type}.yaml", "r") as f:
        return parse_config(yaml.safe_load(f))


def test_time_repeats():
    """Warmup calls happen but aren't timed, and there is one timing per repeat"""

    calls = []
    times = benchmark.time_repeats(lambda: calls.append(1), warmup=3, repeats=5)

    assert len(calls) == 8
    assert len(times) == 5
    assert all(t >= 0 for t in times)


def test_summarise_times():
    summary = benchmark.summarise_times([0.001, 0.003, 0.002])

    assert summary["repeats"] == 3
    assert summary["median_ms"] == pytest.approx(2.0)
    assert summary["mean_ms"] == pytest.approx(2.0)
    assert summary["std_ms"] == pytest.approx(1.0)
    assert summary["min_ms"] == pytest.approx(1.0)
    assert summary["max_ms"] == pytest.approx(3.0)

    # standard deviation needs at least two samples
    assert benchmark.summarise_times([0.001])["std_ms"] is None


def test_model_memory():
    """Parameters and buffers (the causal masks) are counted, at 4 bytes each in fp32"""

    model = TransformerLanguageModel(
        vocab_size=65,
        n_embd=32,
        max_seq_len=8,
        num_heads=4,
        n_layers=2,
        dropout=0.0,
        group_size=1,
    )
    masks = 2 * 8 * 8  # one (max_seq_len, max_seq_len) mask per layer
    expected = (count_parameters(model) + masks) * 4 / 2**20

    assert get_model_memory_mb(model) == pytest.approx(expected)


@pytest.mark.parametrize(("model_type"), ["bigram", "attention", "transformer"])
def test_benchmark(tmp_path, model_type):
    """Every model can be benchmarked, and the results are saved"""

    cfg = get_debug_config(model_type)
    record = benchmark.main(
        cfg, num_tokens=5, warmup=1, repeats=2, results_dir=tmp_path
    )

    tokens = cfg.batch_size * cfg.max_seq_len
    for name in ["training", "evaluation"]:
        assert record[name]["tokens_per_step"] == tokens
        assert record[name]["repeats"] == 2
        assert record[name]["tokens_per_sec"] > 0

    # by default, the shortest and longest prompts
    prompt_lens = [g["prompt_len"] for g in record["generation"]]
    assert prompt_lens == sorted({1, cfg.max_seq_len})
    for g in record["generation"]:
        assert g["num_tokens"] == 5
        assert g["tokens_per_sec"] > 0

    saved = load_runs(tmp_path)
    assert len(saved) == 1
    assert saved[0]["run_id"] == record["run_id"]
    assert saved[0]["parameters"] == record["parameters"]


def test_benchmark_prompt_lens():
    """A prompt longer than the model's context can't be benchmarked"""

    cfg = get_debug_config("transformer")
    record = benchmark.main(cfg, prompt_lens=[2, 4], num_tokens=3, warmup=0, repeats=1)
    assert [g["prompt_len"] for g in record["generation"]] == [2, 4]

    with pytest.raises(AssertionError):
        benchmark.main(cfg, prompt_lens=[cfg.max_seq_len + 1], warmup=0, repeats=1)


def test_benchmark_checkpoint(tmp_path):
    """A trained checkpoint can be benchmarked"""

    cfg = get_debug_config("transformer")
    cfg.iterations = 10
    path = tmp_path / "model.pt"
    model = train.main(cfg, output_ckpt=path)

    record = benchmark.main(cfg, path, num_tokens=3, warmup=0, repeats=1)

    assert record["checkpoint"] == str(path)
    assert record["parameters"] == count_parameters(model)


def test_benchmarks_not_compared(tmp_path):
    """
    Benchmarks are saved in a directory inside the training results, which compare.py
    doesn't load, because benchmarks don't have losses
    """

    assert benchmark.BENCHMARKS_DIR.parent == benchmark.RESULTS_DIR

    cfg = get_debug_config("transformer")
    cfg.iterations = 10
    train.main(cfg, results_dir=tmp_path)
    benchmark.main(
        cfg, num_tokens=3, warmup=0, repeats=1, results_dir=tmp_path / "benchmarks"
    )

    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert "evals" in runs[0]
