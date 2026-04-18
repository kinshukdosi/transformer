Implementing a transformer from scratch <br>
Focusing on understanding and documenting all steps <br>

Resources: <br>
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017) <br>
Andrej Karpathy's ["Let's build GPT from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY) <br>
[minbpe](https://github.com/karpathy/minbpe) <br>
["Layer Normalization"](https://arxiv.org/abs/1607.06450) (2016) <br>
["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)<br>
["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/pdf/2305.13245)

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

Example configs for all models can be found in `configs/`

### Bigram Language Model

A simple model that predicts the next token given only the current token. It is essentially a lookup table with dimensions based on the vocabulary size of the dataset. Each row corresponds to a token and contains scores for the next token.
[Implementation here.](models/bigram.py)

### Attention Language Model

Implemented in [models/attention.py](models/attention.py). This model uses multi-head attention, nothing else.

### Transformer Language Model
Model that implements the full transformer architecture as detailed in "Attention Is All You Need". Implemented in [models/transformer.py](models/transformer.py)

### Language Model Comparison

| Model Type     | Vocab Size | Dataset         | Batch Size | Iterations | Learning Rate | Block Size | Head Size | # Heads | Validation Loss |
| -------------- | ---------- | --------------- | ---------- | ---------- | ------------- | ---------- | --------- | ------- | --------------- |
| Bigram         | 257        | tinyshakespeare | 32         | 25k        | 1e-3 (AdamW)  | —          | —         | —       | ~2.5            |
| Attention      | 257        | tinyshakespeare | 32         | 25k        | 1e-3 (AdamW)  | 16         | 32        | 1       | ~2.4            |
| Attention      | 257        | tinyshakespeare | 32         | 25k        | 1e-3 (AdamW)  | 16         | 32        | 4       | ~2.1            |

