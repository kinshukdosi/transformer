import argparse
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass
from tokenizer import Tokenizer
from data_loader import batch_data
from models.bigram import BigramLanguageModel

supported_models = ["bigram"]
supported_optimizers = ["AdamW"]


@dataclass
class ModelConfig:
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


def parse_config(cfg_path) -> ModelConfig:

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    return ModelConfig(
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


def main(config: ModelConfig):

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
        # block size = 1 because bigram only looks at previous token
        assert config.block_size == 1, "Block size should be 1 for bigram model"
        model = BigramLanguageModel(config.vocab_size)

    if model is None:
        raise TypeError("Model is not set! Aborting")

    optim = None
    if config.optimizer == "AdamW":
        optim = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if optim is None:
        raise TypeError("Optimizer is not set! Aborting")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to .yaml config file"
    )
    args = parser.parse_args()

    cfg_path = args.config
    assert Path.exists(cfg_path), f"{cfg_path} does not exist"
    assert cfg_path.suffix == ".yaml", f"{cfg_path} must be a .yaml file"

    config = parse_config(cfg_path)
    main(config)
