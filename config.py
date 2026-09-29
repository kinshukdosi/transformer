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

# seed that was hard-coded in train.py before it became a config option. used for
# configs and checkpoints that don't specify one
DEFAULT_SEED = 100

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BaseConfig:
    tokenizer: str
    model_type: str
    vocab_size: int
    data_path: Path
    max_seq_len: int  # upper bound on how many tokens a model can attend to at once
    train_split: float
    batch_size: int
    iterations: int
    optimizer: str
    lr: float
    eval_iterations: int
    eval_interval: int
    seed: int


@dataclass
class BigramConfig(BaseConfig):
    pass


@dataclass
class AttentionConfig(BaseConfig):
    n_embd: int
    num_heads: int
    dropout: float
    group_size: int


@dataclass
class TransformerConfig(AttentionConfig):
    n_layers: int


def parse_config(cfg: dict) -> BaseConfig:

    model_type = cfg["model_type"]
    cfg["data_path"] = Path(cfg["data_path"])
    cfg["lr"] = float(cfg["lr"])
    cfg.setdefault("seed", DEFAULT_SEED)

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

    m = torch.load(model, map_location=device)
    cfg_dict = m["config"]
    config = parse_config(cfg_dict)
    return config


def get_model_from_config(config: BaseConfig):

    model = None
    if config.model_type == "bigram":
        assert isinstance(config, BigramConfig)
        # max_seq_len = 1 because bigram only looks at previous token
        assert (
            config.max_seq_len == 1
        ), "Max sequence length should be 1 for bigram model"

        model = BigramLanguageModel(config.vocab_size)
    elif config.model_type == "attention":
        assert isinstance(config, AttentionConfig)
        model = AttentionHeadLanguageModel(
            config.vocab_size,
            config.n_embd,
            config.max_seq_len,
            config.num_heads,
            config.dropout,
            config.group_size,
        )
    elif config.model_type == "transformer":
        assert isinstance(config, TransformerConfig)
        model = TransformerLanguageModel(
            config.vocab_size,
            config.n_embd,
            config.max_seq_len,
            config.num_heads,
            config.n_layers,
            config.dropout,
            config.group_size,
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
