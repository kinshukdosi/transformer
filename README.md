Implementing a transformer from scratch <br>
Focusing on understanding and documenting all steps <br>

Resources: <br>
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017) <br>
Andrej Karpathy's ["Let's build GPT from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY) <br>
[minbpe](https://github.com/karpathy/minbpe)

Requirements: Python 3.12

### Project setup
`pip install -r requirements.txt`

### To run tests
Run `pytest` from the root directory.

### To train a model on dataset
Create a config.yaml file with your desired training parameters <br>
Run `python train.py --config <path_to_config_file>`

### Dataset and tokenization

Uses Karpathy's tinyshakespeare dataset for now. <br>
"Attention Is All You Need" implements byte-pair encoding (as mentioned in 5.1 Training Data and Batching), this project does the same. <br>

## Models
### Bigram language model


Simple model, predicts the next token given only the current token. It's essentially a lookup table with dimensions based on the size of the vocabulary of the dataset. Each row of the lookup table corresponds to one token, and contains the scores for the next token.
[Implementation here.](models/bigram.py) <br>

The default training configuration in [bigram.yaml](config/bigram.yaml) achieves the following results:

| Setting        | Value                    |
|----------------|--------------------------|
| Vocab Size     | 257                      |
| Dataset        | tinyshakespeare          |
| Batch Size     | 32                       |
| Iterations     | 25k                      |
| Learning Rate  | 1e-3 (AdamW)             |

| Result         | Value |
|----------------|-------|
| Loss           | ~2.5  |