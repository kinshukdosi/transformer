"""
Compare training runs side by side. Prints one row per run, with the config settings
that differ between the runs and the main measurements, and warns about differences
that stop the runs being directly comparable
"""

import argparse
import dataclasses
from pathlib import Path
from typing import Optional
from config import parse_config
from results import RESULTS_DIR, load_runs

# shown in the run id already, so it doesn't need its own column
HIDDEN_CONFIG_KEYS = ["experiment"]


def get_config(run: dict) -> dict:
    # parse the saved config again, so that runs saved before a setting existed get
    # the default they actually trained with, rather than looking like they differ
    config = dataclasses.asdict(parse_config(dict(run["config"])))
    config["data_path"] = str(config["data_path"])
    return config


def config_differences(runs: list[dict]) -> list[str]:
    """Config keys that don't have the same value in every run, in config order"""
    configs = [get_config(run) for run in runs]

    keys = []
    for config in configs:
        for key in config:
            if key not in keys and key not in HIDDEN_CONFIG_KEYS:
                keys.append(key)

    # a model without a setting (e.g. bigram has no n_layers) counts as a difference
    return [key for key in keys if len({str(c.get(key)) for c in configs}) > 1]


def comparability_warnings(runs: list[dict]) -> list[str]:
    """Differences that aren't experimental settings but still affect the results"""
    warnings = []

    hashes = {run.get("dataset", {}).get("sha256") for run in runs}
    if None in hashes:
        warnings.append(
            "Some runs were saved before the dataset was recorded, so it can't be "
            "checked that every run used the same data"
        )
    elif len(hashes) > 1:
        warnings.append("Runs used different datasets")

    tokenizers = {
        (run["config"]["tokenizer"], run["config"]["vocab_size"]) for run in runs
    }
    if len(tokenizers) > 1:
        warnings.append(
            "Runs used different tokenizers or vocab sizes. Losses are per token, so "
            "they aren't directly comparable"
        )

    hardware = {(run["system"]["device"], run["system"]["gpu"]) for run in runs}
    if len(hardware) > 1:
        warnings.append(
            "Runs ran on different hardware, so throughput and memory aren't comparable"
        )

    if any(run["git_dirty"] for run in runs):
        warnings.append(
            "Runs marked * had uncommitted changes, so their commit doesn't fully "
            "describe the code"
        )

    return warnings


def format_number(value: Optional[float], fmt: str) -> str:
    return "-" if value is None else format(value, fmt)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    # each column is as wide as its widest cell
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
        for row in [headers, ["-" * width for width in widths], *rows]
    ]
    return "\n".join(line.rstrip() for line in lines)


def compare_runs(runs: list[dict]) -> str:
    """A table with one row per run, followed by any comparability warnings"""
    differences = config_differences(runs)
    headers = [
        "run",
        *differences,
        "params",
        "train loss",
        "val loss",
        "best val loss",
        "best step",
        "tokens/sec",
        "peak MB",
        "commit",
    ]

    rows = []
    for run in runs:
        config = get_config(run)
        final = run["final"]
        best = run["best"]
        training = run["training"]
        rows.append(
            [
                run["run_id"],
                *[str(config.get(key, "-")) for key in differences],
                format_number(run["parameters"], ","),
                format_number(final["train_loss"], ".4f"),
                format_number(final["val_loss"], ".4f"),
                format_number(best["val_loss"], ".4f"),
                str(best["step"]),
                format_number(training["tokens_per_sec"], ",.0f"),
                format_number(training["peak_memory_mb"], ",.0f"),
                run["git_commit"][:7] + ("*" if run["git_dirty"] else ""),
            ]
        )

    warnings = [f"Warning: {warning}" for warning in comparability_warnings(runs)]
    return "\n".join([format_table(headers, rows), "", *warnings]).rstrip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dirs",
        type=Path,
        nargs="*",
        default=[RESULTS_DIR],
        help="Directories of run results to compare (default: results/dev)",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        help="Only compare runs whose experiment name contains this",
    )
    args = parser.parse_args()

    runs = [run for results_dir in args.results_dirs for run in load_runs(results_dir)]
    if args.experiment is not None:
        runs = [run for run in runs if args.experiment in (run.get("experiment") or "")]

    if len(runs) == 0:
        print("No runs found")
    else:
        print(compare_runs(runs))
