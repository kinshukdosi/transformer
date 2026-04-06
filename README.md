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

### To save training progress to a checkpoint
`python train.py --config <path_to_config_file> --save` saves to `checkpoint.pt` by default <br>
`python train.py --config <path_to_config_file> --save -o <output_checkpoint_path>` <br>

### To load from a checkpoint
`python train.py --load` loads from `checkpoint.pt` by default <br>
`python train.py --load -i <input_checkpoint_path>` <br>

### Dataset and tokenization

Uses Karpathy's tinyshakespeare dataset for now. <br>
"Attention Is All You Need" implements byte-pair encoding (as mentioned in 5.1 Training Data and Batching), this project does the same. <br>

## Models

### Bigram Language Model

A simple model that predicts the next token given only the current token. It is essentially a lookup table with dimensions based on the vocabulary size of the dataset. Each row corresponds to a token and contains scores for the next token.
[Implementation here.](models/bigram.py)

#### Configuration

| Setting       | Value           |
| ------------- | --------------- |
| Vocab Size    | 257             |
| Dataset       | tinyshakespeare |
| Batch Size    | 32              |
| Iterations    | 25k             |
| Learning Rate | 1e-3 (AdamW)    |

#### Results

| Metric          | Value |
| --------------- | ----- |
| Validation Loss | ~2.5  |

---

### Attention Model

Implemented in [models/attention.py](models/attention.py). This model uses multi-head attention, nothing else. The configuration I used for the multi-head training can be found [here](config/attention.yaml)

#### Configuration

| Setting         | Value           |
| --------------- | --------------- |
| Vocab Size      | 257             |
| Dataset         | tinyshakespeare |
| Batch Size      | 32              |
| Iterations      | 25k             |
| Learning Rate   | 1e-3 (AdamW)    |
| Block Size      | 16              |
| Head Size       | 32              |
| Number of Heads | 1 / 4           |

#### Results

| Model Variant | Validation Loss |
| ------------- | --------------- |
| Single Head   | ~2.4            |
| 4 Heads       | ~2.1            |
