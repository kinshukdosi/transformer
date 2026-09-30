import torch
import argparse
import pathlib
from pathlib import Path
from typing import Optional
from data_loader import split_text
from config import (
    get_config_from_pytorch_model,
    get_model_from_config,
    get_tokenizer_from_config,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.serialization.add_safe_globals([pathlib.PosixPath])


def main(model_path: Path, prompt: Optional[str]):

    config = get_config_from_pytorch_model(model_path)
    model = get_model_from_config(config)

    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )  # load trained weights

    model.eval()  # switch to inference
    tokenizer = get_tokenizer_from_config(config)

    with open(config.data_path, "r") as f:
        text = f.read()

    # retrain the tokenizer exactly as train.py did, on the training split only
    train_text, _ = split_text(text, config.train_split)
    tokenizer.train(train_text, config.vocab_size)

    if prompt is not None:
        seed = tokenizer.encode(prompt)
        context = torch.tensor([seed], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # disable gradients for memory and speed efficiency
    with torch.no_grad():
        output = model.generate(context, num_tokens=500)

    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, help="Path to .pt model file")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    args = parser.parse_args()

    model_path = args.model
    assert Path.exists(model_path), f"{model_path} doesn't exist"
    assert model_path.suffix == ".pt", "model file must be .pt file"

    main(model_path, args.prompt)
