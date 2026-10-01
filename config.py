import torch
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from tokenizer import BPETokenizer, SimpleTokenizer
from models.bigram import BigramLanguageModel
from models.attention import AttentionHeadLanguageModel
from models.transformer import TransformerLanguageModel

# any of the language models. they all have the same forward() and generate()
LanguageModel = Union[
    BigramLanguageModel, AttentionHeadLanguageModel, TransformerLanguageModel
]

supported_tokenizers = ["bpe", "simple"]
supported_models = ["bigram", "attention", "transformer"]
supported_optimizers = ["AdamW"]

# seed that was hard-coded in train.py before it became a config option. used for
# configs and checkpoints that don't specify one
DEFAULT_SEED = 100

# PyTorch's AdamW defaults
DEFAULT_OPTIMIZER_SETTINGS = {
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.01,
}

# learning rate schedule and gradient clipping are off unless they're set in the config
DEFAULT_SCHEDULE_SETTINGS = {
    "warmup_iters": 0,
    "lr_decay_iters": None,
    "min_lr": None,
    "grad_clip": None,
}

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
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    warmup_iters: int  # steps to increase the learning rate linearly up to lr
    lr_decay_iters: Optional[int]  # step at which cosine decay reaches min_lr
    min_lr: Optional[float]
    grad_clip: Optional[float]  # max norm of all gradients together
    eval_iterations: int
    eval_interval: int
    seed: int
    experiment: Optional[str]  # name of the experiment the run belongs to


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
    cfg.setdefault("seed", DEFAULT_SEED)
    cfg.setdefault("experiment", None)
    for key, value in {
        **DEFAULT_OPTIMIZER_SETTINGS,
        **DEFAULT_SCHEDULE_SETTINGS,
    }.items():
        cfg.setdefault(key, value)

    # yaml reads numbers like 1e-3 as strings, because it expects a decimal point
    for key in ["lr", *DEFAULT_OPTIMIZER_SETTINGS, "min_lr", "grad_clip"]:
        if cfg[key] is not None:
            cfg[key] = float(cfg[key])

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


def get_model_from_config(config: BaseConfig) -> LanguageModel:

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


def get_optimizer_from_config(config: BaseConfig, model: torch.nn.Module):
    # weight decay pulls weights towards zero, which stops weight matrices and
    # embeddings growing large to overfit. biases and LayerNorm weights and biases (the
    # 1D parameters) only shift and scale activations, and pulling LayerNorm weights
    # towards zero would just shrink the normalized outputs, so they aren't decayed
    params = list(model.parameters())
    param_groups = [
        {
            "params": [p for p in params if p.dim() >= 2],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]

    if config.optimizer == "AdamW":
        return torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    raise TypeError("Optimizer is not set! Aborting")


def get_tokenizer_from_config(config: BaseConfig):
    if config.tokenizer == "bpe":
        return BPETokenizer()
    elif config.tokenizer == "simple":
        return SimpleTokenizer()
    raise NameError("Tokenizer not found")
