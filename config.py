import torch
from pathlib import Path
from dataclasses import dataclass
from tokenizer import BPETokenizer, SimpleTokenizer
from models.bigram import BigramLanguageModel
from models.attention import AttentionHeadLanguageModel
from models.transformer import TransformerLanguageModel

supported_tokenizers = ["bpe", "simple"]
supported_models = ["bigram", "attention", "transformer"]
supported_optimizers = ["AdamW"]

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BaseConfig:
    tokenizer: str
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
    dropout: float


@dataclass
class TransformerConfig(AttentionConfig):
    n_layers: int


def parse_config(cfg: dict) -> BaseConfig:

    model_type = cfg["model_type"]
    cfg["data_path"] = Path(cfg["data_path"])
    cfg["lr"] = float(cfg["lr"])

    if model_type == "bigram":
        return BigramConfig(**cfg)
    elif model_type == "attention" or model_type == "transformer":
        cfg["dropout"] = float(cfg["dropout"])
        if model_type == "attention":
            return AttentionConfig(**cfg)
        else:  # transformer
            return TransformerConfig(**cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_config_from_pytorch_model(model: Path):
    assert model.suffix == ".pt", f"{model} must be a .pt file"

    m = torch.load(model)
    cfg_dict = m["config"]
    config = parse_config(cfg_dict)
    return config


def get_model_from_config(config: BaseConfig):

    model = None
    if config.model_type == "bigram":
        assert isinstance(config, BigramConfig)
        # block size = 1 because bigram only looks at previous token
        assert config.block_size == 1, "Block size should be 1 for bigram model"
        model = BigramLanguageModel(config.vocab_size)
    elif config.model_type == "attention":
        assert isinstance(config, AttentionConfig)
        model = AttentionHeadLanguageModel(
            config.vocab_size,
            config.n_embd,
            config.block_size,
            config.num_heads,
            config.dropout,
        )
    elif config.model_type == "transformer":
        assert isinstance(config, TransformerConfig)
        model = TransformerLanguageModel(
            config.vocab_size,
            config.n_embd,
            config.block_size,
            config.num_heads,
            config.n_layers,
            config.dropout,
        )

    if model is None:
        raise TypeError("Model is not set! Aborting")

    model.to(device)
    return model


def get_tokenizer_from_config(config: BaseConfig):
    if config.tokenizer == "bpe":
        return BPETokenizer()
    elif config.tokenizer == "simple":
        return SimpleTokenizer()
    raise NameError("Tokenizer not found")
