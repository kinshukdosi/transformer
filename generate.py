import torch
import argparse
import pathlib
from pathlib import Path
from config import (
    get_config_from_pytorch_model,
    get_model_from_config,
    get_tokenizer_from_config,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.serialization.add_safe_globals([pathlib.PosixPath])


def main(model_path: Path):

    config = get_config_from_pytorch_model(model_path)
    model = get_model_from_config(config)

    model.eval()  # switch to inference
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # disable gradients for memory and speed efficiency
    with torch.no_grad():
        output = model.generate(context, num_tokens=500)

    tokenizer = get_tokenizer_from_config(config)
    with open(config.data_path, "r") as f:
        text = f.read()
    tokenizer.train(text, config.vocab_size)

    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, help="Path to .pt model file")
    args = parser.parse_args()

    model_path = args.model
    assert Path.exists(model_path), f"{model_path} doesn't exist"
    assert model_path.suffix == ".pt", "model file must be .pt file"

    main(model_path)
