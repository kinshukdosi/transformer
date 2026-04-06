import argparse
import yaml
import torch
import pathlib
import dataclasses
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from tokenizer import Tokenizer
from data_loader import batch_data
from models.bigram import BigramLanguageModel
from models.attention import AttentionHeadLanguageModel

supported_models = ["bigram", "attention"]
supported_optimizers = ["AdamW"]

# so that we can load/save checkpoints that include configs with pathlib.Path
torch.serialization.add_safe_globals([pathlib.PosixPath])


@dataclass
class BaseConfig:
    model_type: str
    vocab_size: int
    data_path: Path
    block_size: int
    train_split: float
    batch_size: int
    iterations: int
    optimizer: str
    lr: float
    eval_iterations: int
    eval_interval: int


@dataclass
class BigramConfig(BaseConfig):
    pass


@dataclass
class AttentionConfig(BaseConfig):
    n_embd: int
    num_heads: int


def parse_config(cfg: dict) -> BaseConfig:

    model_type = cfg["model_type"]
    cfg["data_path"] = Path(cfg["data_path"])
    cfg["lr"] = float(cfg["lr"])

    if model_type == "bigram":
        return BigramConfig(**cfg)
    elif model_type == "attention":
        return AttentionConfig(**cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main(
    config: BaseConfig,
    output_ckpt: Optional[Path] = None,
    input_ckpt: Optional[Path] = None,
):

    assert config.model_type in supported_models, "Model not supported"
    assert config.vocab_size > 256, "Vocab size must be greater than 256"
    assert Path.exists(config.data_path), "Dataset doesn't exist"
    assert config.data_path.suffix == ".txt", "Dataset should be .txt file"
    assert config.train_split <= 1.0, "Training split should be less than 1"
    assert config.batch_size >= 1, "Batch size should be >= 1"
    assert config.iterations >= 1, "Number of training iterations should be >= 1"
    assert config.optimizer in supported_optimizers, "Optimizer not supported"

    tokenizer = Tokenizer()
    with open(config.data_path, "r") as f:
        text = f.read()
    tokenizer.train(text, config.vocab_size)
    text_encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = len(text_encoded)
    train_data = text_encoded[: int(n * config.train_split)]
    val_data = text_encoded[int(n * config.train_split) :]

    model = None
    if config.model_type == "bigram":
        assert isinstance(config, BigramConfig)
        # block size = 1 because bigram only looks at previous token
        assert config.block_size == 1, "Block size should be 1 for bigram model"
        model = BigramLanguageModel(config.vocab_size)
    elif config.model_type == "attention":
        assert isinstance(config, AttentionConfig)
        model = AttentionHeadLanguageModel(
            config.vocab_size, config.n_embd, config.block_size, config.num_heads
        )

    if model is None:
        raise TypeError("Model is not set! Aborting")

    optim = None
    if config.optimizer == "AdamW":
        optim = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if optim is None:
        raise TypeError("Optimizer is not set! Aborting")

    if input_ckpt is not None:
        print(f"Starting training from checkpoint: {input_ckpt}")
        checkpoint = torch.load(input_ckpt)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])

    # efficiency, telling pytorch we will never run backpropagation here
    @torch.no_grad()
    def model_eval(data: torch.Tensor):
        model.eval()
        losses = torch.zeros(config.eval_iterations)
        for i in range(config.eval_iterations):
            inputs, targets = batch_data(data, config.batch_size, config.block_size)
            _, loss = model(inputs, targets)  # logits don't matter here
            losses[i] = loss.item()
        model.train()
        return losses.mean()

    for i in range(config.iterations):
        if i % config.eval_interval == 0:
            train_loss = model_eval(train_data)
            val_loss = model_eval(val_data)
            print(
                f"Step {i}: Training loss = {train_loss}, Validation loss = {val_loss}"
            )

        inputs, targets = batch_data(train_data, config.batch_size, config.block_size)
        _, loss = model(inputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

    if output_ckpt is not None:
        checkpoint = {
            "config": dataclasses.asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        }
        print(f"Saving checkpoint: {args.save}")
        torch.save(checkpoint, output_ckpt)


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
    else:
        assert (
            args.load is not None
        ), "You need to either pass a config file or load from a checkpoint"
        input_ckpt = Path(args.load)
        assert input_ckpt.suffix == ".pt", f"{input_ckpt} must be a .pt file"

        checkpoint = torch.load(input_ckpt)
        cfg_dict = checkpoint["config"]

    config = parse_config(cfg_dict)
    if args.iterations is not None:  # override number of iterations in config
        config.iterations = args.iterations

    if args.save is not None:
        output_ckpt = Path(args.save)
        assert output_ckpt.suffix == ".pt", f"{output_ckpt} must be a .pt file"

    main(config, output_ckpt, input_ckpt)
