import argparse
import yaml
import torch
import pathlib
import dataclasses
from pathlib import Path
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


def parse_config(cfg_path) -> BaseConfig:

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_kwargs = dict(
        model_type=cfg["model_type"],
        vocab_size=cfg["vocab_size"],
        data_path=Path(cfg["dataset"]),
        block_size=cfg["block_size"],
        train_split=cfg["train_split"],
        batch_size=cfg["batch_size"],
        iterations=cfg["iterations"],
        optimizer=cfg["optimizer"],
        lr=float(cfg["learning_rate"]),
        eval_iterations=cfg["eval_iterations"],
        eval_interval=cfg["eval_interval"],
    )

    model_type = cfg["model_type"]

    if model_type == "bigram":
        return BigramConfig(**base_kwargs)
    elif model_type == "attention":
        return AttentionConfig(
            **base_kwargs,
            n_embd=cfg["n_embd"],
            num_heads=cfg["num_heads"],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main(
    config: BaseConfig,
    output_ckpt: Path = Path("checkpoint.pt"),
    save: bool = False,
    load: bool = False,
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

    if load:
        checkpoint = torch.load(input_ckpt_path)
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

    if save:
        checkpoint = {
            "config": dataclasses.asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        }
        torch.save(checkpoint, output_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, help="Path to .yaml config file")
    parser.add_argument("--save", "-s", action="store_true", help="Save training state")
    parser.add_argument("--load", "-l", action="store_true", help="Load training state")
    parser.add_argument("--iterations", type=int, help="Override number of iterations")
    parser.add_argument(
        "--output", "-o", default="checkpoint.pt", help="Output checkpoint file path"
    )
    parser.add_argument(
        "--input", "-i", default="checkpoint.pt", help="Input checkpoint file path"
    )
    args = parser.parse_args()

    output_ckpt_path = Path(args.output)
    input_ckpt_path = Path(args.input)

    cfg_path = args.config
    if cfg_path is not None:
        assert Path.exists(cfg_path), f"{cfg_path} does not exist"
        assert cfg_path.suffix == ".yaml", f"{cfg_path} must be a .yaml file"
        config = parse_config(cfg_path)
    else:
        assert (
            args.load is not None
        ), "You need to either pass a config file or load from a checkpoint"
        assert (
            output_ckpt_path.suffix == ".pt"
        ), f"{output_ckpt_path} must be a .pt file"
        assert input_ckpt_path.suffix == ".pt", f"{input_ckpt_path} must be a .pt file"

        checkpoint = torch.load(input_ckpt_path)
        cfg_dict = checkpoint["config"]
        if cfg_dict["model_type"] == "bigram":
            config = BigramConfig(**cfg_dict)
        elif cfg_dict["model_type"] == "attention":
            config = AttentionConfig(**cfg_dict)
        else:
            raise NotImplementedError("Model type not yet implemented")

    if args.iterations is not None:  # override number of iterations in config
        config.iterations = args.iterations
    main(config, output_ckpt_path, args.save, args.load)
