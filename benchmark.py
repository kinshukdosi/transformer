"""
Benchmark a model's speed and memory, separately from its quality. Measures training
and evaluation throughput, generation latency and peak GPU memory.

Inputs are random tokens and timings don't depend on the weights, so a benchmark only
needs a config. The dataset, tokenizer and data loading aren't part of the measurement.
"""

import argparse
import statistics
import time
import torch
import yaml
from pathlib import Path
from typing import Callable, Optional
from compare import format_number, format_table
from train import synchronize
from results import (
    RESULTS_DIR,
    new_record,
    save_run,
    get_model_memory_mb,
    get_peak_memory_mb,
)
from config import (
    device,
    BaseConfig,
    LanguageModel,
    parse_config,
    get_config_from_pytorch_model,
    get_model_from_config,
    get_optimizer_from_config,
)

# inside results/dev so it isn't tracked by git, and in its own directory so that
# compare.py doesn't mistake benchmarks for training runs
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"

DEFAULT_WARMUP = 5
DEFAULT_REPEATS = 20
DEFAULT_NUM_TOKENS = 200


def time_repeats(fn: Callable[[], object], warmup: int, repeats: int) -> list[float]:
    """Call fn warmup times untimed, then time it repeats times. Returns seconds"""

    # the first calls are slower: CUDA initializes, kernels are chosen and the memory
    # allocator grows its cache. we don't want any of that in the measurement
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        fn()
        synchronize()
        times.append(time.perf_counter() - start)
    return times


def summarise_times(times: list[float]) -> dict:
    """Median is used for throughput because it isn't skewed by the odd slow repeat"""
    times_ms = [t * 1000 for t in times]
    return {
        "repeats": len(times_ms),
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else None,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def reset_peak_memory():
    # peak memory is measured separately for each benchmark
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def random_batch(config: BaseConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of random tokens, the same shape train.py uses"""
    shape = (config.batch_size, config.max_seq_len)
    inputs = torch.randint(config.vocab_size, shape, device=device)
    targets = torch.randint(config.vocab_size, shape, device=device)
    return inputs, targets


def benchmark_training(
    model: torch.nn.Module, config: BaseConfig, warmup: int, repeats: int
) -> dict:
    """Time one optimizer step: forward, backward and AdamW update"""
    optim = get_optimizer_from_config(config, model)
    inputs, targets = random_batch(config)
    model.train()

    def step():
        _, loss = model(inputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

    reset_peak_memory()
    timings = summarise_times(time_repeats(step, warmup, repeats))

    tokens = config.batch_size * config.max_seq_len
    return {
        "batch_size": config.batch_size,
        "seq_len": config.max_seq_len,
        "tokens_per_step": tokens,
        **timings,
        "tokens_per_sec": tokens / (timings["median_ms"] / 1000),
        "peak_memory_mb": get_peak_memory_mb(),
    }


def benchmark_evaluation(
    model: torch.nn.Module, config: BaseConfig, warmup: int, repeats: int
) -> dict:
    """Time one forward pass and loss, the way train.py evaluates"""
    inputs, targets = random_batch(config)
    model.eval()

    @torch.no_grad()
    def step():
        model(inputs, targets)

    reset_peak_memory()
    timings = summarise_times(time_repeats(step, warmup, repeats))

    tokens = config.batch_size * config.max_seq_len
    return {
        "batch_size": config.batch_size,
        "seq_len": config.max_seq_len,
        "tokens_per_step": tokens,
        **timings,
        "tokens_per_sec": tokens / (timings["median_ms"] / 1000),
        "peak_memory_mb": get_peak_memory_mb(),
    }


def benchmark_generation(
    model: LanguageModel,
    config: BaseConfig,
    prompt_len: int,
    num_tokens: int,
    warmup: int,
    repeats: int,
) -> dict:
    """
    Time generating num_tokens new tokens for one sequence after a prompt of
    prompt_len tokens, and separately the latency until the first new token. There
    is no KV cache yet, so every new token runs a forward pass over the whole context
    """
    assert 1 <= prompt_len <= config.max_seq_len, "prompt_len must fit the context"
    assert num_tokens >= 1, "num_tokens should be >= 1"

    prompt = torch.randint(config.vocab_size, (1, prompt_len), device=device)
    model.eval()

    @torch.no_grad()
    def generate(n: int):
        model.generate(prompt, num_tokens=n)

    reset_peak_memory()
    first_token = summarise_times(time_repeats(lambda: generate(1), warmup, repeats))
    total = summarise_times(time_repeats(lambda: generate(num_tokens), warmup, repeats))

    return {
        "prompt_len": prompt_len,
        "num_tokens": num_tokens,
        "first_token_ms": first_token["median_ms"],
        **total,
        "ms_per_token": total["median_ms"] / num_tokens,
        "tokens_per_sec": num_tokens / (total["median_ms"] / 1000),
        "peak_memory_mb": get_peak_memory_mb(),
    }


def main(
    config: BaseConfig,
    input_ckpt: Optional[Path] = None,
    prompt_lens: Optional[list[int]] = None,
    num_tokens: int = DEFAULT_NUM_TOKENS,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    results_dir: Optional[Path] = None,
) -> dict:

    assert warmup >= 0, "warmup should be >= 0"
    assert repeats >= 1, "repeats should be >= 1"

    # by default, the shortest and longest prompts the model can take
    if prompt_lens is None:
        prompt_lens = sorted({1, config.max_seq_len})

    # seeded so that the random inputs are the same every time
    torch.manual_seed(config.seed)

    model = get_model_from_config(config)
    if input_ckpt is not None:
        # timings don't depend on the weights, but load them so the benchmark is of
        # exactly the model that was trained
        checkpoint = torch.load(input_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    record = new_record(config, model)
    record["checkpoint"] = str(input_ckpt) if input_ckpt is not None else None
    record["benchmark"] = {"warmup": warmup, "repeats": repeats}
    record["model_memory_mb"] = get_model_memory_mb(model)

    # training runs last. it leaves gradients and optimizer state in memory, which
    # would otherwise count towards the peak memory of the other benchmarks
    record["evaluation"] = benchmark_evaluation(model, config, warmup, repeats)
    record["generation"] = [
        benchmark_generation(model, config, prompt_len, num_tokens, warmup, repeats)
        for prompt_len in prompt_lens
    ]
    record["training"] = benchmark_training(model, config, warmup, repeats)

    print(format_benchmark(record))
    if results_dir is not None:
        print(f"\nSaving results: {save_run(record, results_dir)}")

    return record


def format_benchmark(record: dict) -> str:
    """A table with one row per benchmark"""
    headers = ["benchmark", "tokens", "median ms", "std ms", "tokens/sec", "peak MB"]

    def row(name: str, tokens: int, result: dict) -> list[str]:
        return [
            name,
            str(tokens),
            format_number(result["median_ms"], ".2f"),
            format_number(result["std_ms"], ".2f"),
            format_number(result["tokens_per_sec"], ",.0f"),
            format_number(result["peak_memory_mb"], ",.0f"),
        ]

    training = record["training"]
    evaluation = record["evaluation"]
    rows = [
        row("training step", training["tokens_per_step"], training),
        row("evaluation step", evaluation["tokens_per_step"], evaluation),
        *[
            row(f"generate (prompt {g['prompt_len']})", g["num_tokens"], g)
            for g in record["generation"]
        ],
    ]

    system = record["system"]
    lines = [
        f"{record['run_id']}  ({system['gpu'] or system['device']}, "
        f"{record['precision']})",
        f"parameters: {record['parameters']:,}  "
        f"model memory: {record['model_memory_mb']:.1f} MB",
        "",
        format_table(headers, rows),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, help="Path to .yaml config file")
    parser.add_argument("--model", "-m", type=Path, help="Path to .pt checkpoint")
    parser.add_argument(
        "--prompt-lens",
        type=int,
        nargs="+",
        help="Prompt lengths to benchmark generation with (default: 1 and max_seq_len)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=DEFAULT_NUM_TOKENS,
        help=f"Tokens to generate (default: {DEFAULT_NUM_TOKENS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Untimed calls before each benchmark (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Timed calls for each benchmark (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BENCHMARKS_DIR,
        help="Directory to save results to (default: results/dev/benchmarks)",
    )
    args = parser.parse_args()

    assert (args.config is None) != (
        args.model is None
    ), "Pass either a config file or a checkpoint"

    input_ckpt = None
    if args.config is not None:
        assert Path.exists(args.config), f"{args.config} does not exist"
        assert args.config.suffix == ".yaml", f"{args.config} must be a .yaml file"
        with open(args.config, "r") as f:
            config = parse_config(yaml.safe_load(f))
    else:
        assert Path.exists(args.model), f"{args.model} does not exist"
        input_ckpt = args.model
        config = get_config_from_pytorch_model(input_ckpt)

    main(
        config,
        input_ckpt,
        args.prompt_lens,
        args.num_tokens,
        args.warmup,
        args.repeats,
        args.results_dir,
    )
