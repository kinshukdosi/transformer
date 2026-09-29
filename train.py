import argparse
import yaml
import math
import time
import torch
import pathlib
import dataclasses
from pathlib import Path
from typing import Optional
from data_loader import batch_data
from results import RESULTS_DIR, new_run, save_run, get_peak_memory_mb
from config import (
    device,
    BaseConfig,
    parse_config,
    supported_tokenizers,
    supported_models,
    supported_optimizers,
    get_config_from_pytorch_model,
    get_model_from_config,
    get_tokenizer_from_config,
)


# so that we can load/save checkpoints that include configs with pathlib.Path
torch.serialization.add_safe_globals([pathlib.PosixPath])


def get_rng_state() -> dict:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(rng_state: dict):
    # map_location may have moved these to the GPU, but PyTorch expects CPU tensors
    torch.set_rng_state(rng_state["cpu"].cpu())
    if rng_state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])


def synchronize():
    # CUDA kernels run asynchronously, so wait for them to finish before reading a
    # timer, otherwise we only measure how long it took to launch them
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(
    config: BaseConfig,
    output_ckpt: Optional[Path] = None,
    input_ckpt: Optional[Path] = None,
    results_dir: Optional[Path] = None,
):

    assert config.tokenizer in supported_tokenizers, "Tokenizer not supported"
    assert config.model_type in supported_models, "Model not supported"
    assert Path.exists(config.data_path), "Dataset doesn't exist"
    assert config.data_path.suffix == ".txt", "Dataset should be .txt file"
    assert config.train_split <= 1.0, "Training split should be less than 1"
    assert config.batch_size >= 1, "Batch size should be >= 1"
    assert config.iterations >= 1, "Number of training iterations should be >= 1"
    assert config.optimizer in supported_optimizers, "Optimizer not supported"

    # seed here rather than at import, so that every call to main() is reproducible.
    # this seeds the CPU and all CUDA devices
    torch.manual_seed(config.seed)

    tokenizer = get_tokenizer_from_config(config)
    with open(config.data_path, "r") as f:
        text = f.read()
    tokenizer.train(text, config.vocab_size)

    if config.vocab_size != tokenizer.vocab_size:
        print("Config/Tokenizer vocab size mismatch! Using tokenizer vocab size")
        config.vocab_size = tokenizer.vocab_size

    text_encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = len(text_encoded)
    train_data = text_encoded[: int(n * config.train_split)]
    val_data = text_encoded[int(n * config.train_split) :]

    model = get_model_from_config(config)

    optim = None
    if config.optimizer == "AdamW":
        optim = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if optim is None:
        raise TypeError("Optimizer is not set! Aborting")

    # config.iterations is the total number of steps, so a resumed run continues from
    # the step it was saved at. checkpoints saved before the step and RNG state were
    # recorded start again from step 0
    start_step = 0
    if input_ckpt is not None:
        print(f"Starting training from checkpoint: {input_ckpt}")
        checkpoint = torch.load(input_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0)
        if "rng_state" in checkpoint:
            set_rng_state(checkpoint["rng_state"])

    run = new_run(config, model)
    run["resumed_from"] = str(input_ckpt) if input_ckpt is not None else None

    # efficiency, telling pytorch we will never run backpropagation here
    @torch.no_grad()
    def model_eval(data: torch.Tensor):
        model.eval()
        losses = torch.zeros(config.eval_iterations)
        for i in range(config.eval_iterations):
            inputs, targets = batch_data(data, config.batch_size, config.max_seq_len)
            _, loss = model(inputs, targets)  # logits don't matter here
            losses[i] = loss.item()
        model.train()
        return losses.mean()

    def evaluate(step: int) -> float:
        """Evaluate on both splits, record the losses and return the time taken"""
        synchronize()
        start = time.perf_counter()
        train_loss = model_eval(train_data).item()
        val_loss = model_eval(val_data).item()
        synchronize()

        print(
            f"Step {step}: Training loss = {train_loss}, Validation loss = {val_loss}"
        )
        run["evals"].append(
            {"step": step, "train_loss": train_loss, "val_loss": val_loss}
        )
        return time.perf_counter() - start

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # evaluation time is subtracted so that throughput only measures training steps
    eval_time = 0.0
    synchronize()
    start = time.perf_counter()

    for step in range(start_step, config.iterations):
        if step % config.eval_interval == 0:
            eval_time += evaluate(step)

        inputs, targets = batch_data(train_data, config.batch_size, config.max_seq_len)
        _, loss = model(inputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

    synchronize()
    train_time = time.perf_counter() - start - eval_time

    # save the RNG state before the final evaluation. a resumed run then evaluates at
    # this step with the same random batches, exactly like an uninterrupted run would
    rng_state = get_rng_state()
    evaluate(config.iterations)

    steps = config.iterations - start_step
    tokens = steps * config.batch_size * config.max_seq_len
    run["training"] = {
        "start_step": start_step,
        "steps": steps,
        "tokens": tokens,
        "time_s": train_time,
        "tokens_per_sec": tokens / train_time if train_time > 0 else None,
        "peak_memory_mb": get_peak_memory_mb(),
    }
    run["final"] = {
        **run["evals"][-1],
        # per-token perplexity, so only comparable between runs with the same tokenizer
        "val_perplexity": math.exp(run["evals"][-1]["val_loss"]),
    }

    if output_ckpt is not None:
        checkpoint = {
            "config": dataclasses.asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "step": config.iterations,
            "rng_state": rng_state,
            "run_id": run["run_id"],
        }
        print(f"Saving checkpoint: {output_ckpt}")
        torch.save(checkpoint, output_ckpt)
        run["checkpoint"] = str(output_ckpt)

    if results_dir is not None:
        print(f"Saving results: {save_run(run, results_dir)}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, help="Path to .yaml config file")
    parser.add_argument("--iterations", type=int, help="Override number of iterations")
    parser.add_argument(
        "--save",
        "-s",
        nargs="?",
        const="checkpoint.pt",
        default=None,
        help="Save training state",
    )
    parser.add_argument(
        "--load",
        "-l",
        nargs="?",
        const="checkpoint.pt",
        default=None,
        help="Load training state",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to save run results to (default: results/dev, not tracked)",
    )

    args = parser.parse_args()

    cfg_path = args.config
    input_ckpt = None
    output_ckpt = None
    if cfg_path is not None:
        assert Path.exists(cfg_path), f"{cfg_path} does not exist"
        assert cfg_path.suffix == ".yaml", f"{cfg_path} must be a .yaml file"
        with open(cfg_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        print(f"Starting training from config file: {cfg_path}")
        if args.load is not None:
            print(
                "You have set passed a config file and also set --load! Loading from "
                "checkpoint will be ignored."
            )
        config = parse_config(cfg_dict)
    else:
        assert (
            args.load is not None
        ), "You need to either pass a config file or load from a checkpoint"
        input_ckpt = Path(args.load)
        config = get_config_from_pytorch_model(input_ckpt)

    if args.iterations is not None:  # override number of iterations in config
        config.iterations = args.iterations

    if args.save is not None:
        output_ckpt = Path(args.save)
        assert output_ckpt.suffix == ".pt", f"{output_ckpt} must be a .pt file"

    main(config, output_ckpt, input_ckpt, args.results_dir)
