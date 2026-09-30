"""
Structured results for experiment runs. Every run writes one JSON file containing the
config, the git commit, the software/hardware it ran on and the measurements taken, so
runs can be compared programmatically and reproduced later.
"""

import git
import json
import hashlib
import uuid
import torch
import platform
import dataclasses
from pathlib import Path
from datetime import datetime
from typing import Optional
from config import BaseConfig

REPO_DIR = Path(__file__).parent
# development and debugging runs go to results/dev, which isn't tracked by git. only
# research runs are committed, by saving them to another directory with --results-dir
RESULTS_DIR = REPO_DIR / "results" / "dev"


def get_commit_hash() -> str:
    return git.Repo(REPO_DIR).head.object.hexsha


def is_repo_dirty() -> bool:
    # uncommitted changes to tracked files mean the commit hash alone doesn't describe
    # the code that produced a result
    return git.Repo(REPO_DIR).is_dirty()


def get_run_id(config: BaseConfig) -> str:
    # timestamp first so that run files sort chronologically. the random suffix stops
    # two runs started in the same second from overwriting each other
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = config.experiment if config.experiment is not None else config.model_type
    return f"{timestamp}-{name}-{uuid.uuid4().hex[:6]}"


def get_system_info() -> dict:
    """Software and hardware versions. Results are only comparable on the same setup"""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": None,
        "gpu_memory_mb": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_mb"] = props.total_memory / 2**20
    return info


def get_dataset_info(data_path: Path) -> dict:
    """Identify the dataset by a hash of its contents, not just its file name"""
    with open(data_path, "rb") as f:
        data = f.read()
    return {
        "path": str(data_path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def new_run(config: BaseConfig, model: torch.nn.Module) -> dict:
    """Start a run record. Measurements are added to it during training"""
    return {
        "run_id": get_run_id(config),
        "experiment": config.experiment,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_commit_hash(),
        "git_dirty": is_repo_dirty(),
        "system": get_system_info(),
        "config": dataclasses.asdict(config),
        "parameters": count_parameters(model),
        "precision": "fp32",
        "resumed_from": None,
        "evals": [],  # list of {"step", "train_loss", "val_loss"}
    }


def save_run(run: dict, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{run['run_id']}.json"

    # default=str converts values JSON can't handle, like pathlib.Path in the config
    with open(path, "w") as f:
        json.dump(run, f, indent=2, default=str)

    return path


def load_runs(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Load every run in results_dir, oldest first"""
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r") as f:
            runs.append(json.load(f))
    return runs


def get_peak_memory_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**20
