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
Run `python train.py --config <path_to_config_file>` <br> `--iterations <n>` overrides the number of iterations in the config <br>

Training is seeded with the `seed` value in the config, so running the same config twice gives the same model.

Evaluation uses the same fixed, non-overlapping windows every time: up to `eval_iterations × batch_size` windows spread evenly across each split. It doesn't use the random number generator, so changing the evaluation settings doesn't change training.

### To save training progress to a checkpoint
`python train.py --config <path_to_config_file> --save` saves to `checkpoint.pt` by default <br>
`python train.py --config <path_to_config_file> --save <output_checkpoint_path>` <br>

Checkpoints contain the config, model and optimizer state, the trained tokenizer, the step reached and the random number generator state, so generating text doesn't need the dataset.

### To load from a checkpoint
`python train.py --load` loads from `checkpoint.pt` by default <br>
`python train.py --load <input_checkpoint_path>` <br>
`python train.py --load <input_checkpoint_path> --iterations 20000` continues training up to step 20000 <br>

`iterations` is the total number of training steps, so a loaded checkpoint continues from the step it was saved at. Resuming gives exactly the same result as training without stopping. `--load` is ignored if `--config` is also passed.

### To generate text
`python generate.py --model <checkpoint_path> --prompt "ROMEO:"` <br>

### Run results
Every training run saves a JSON file with:
- run id, experiment name, git commit, and whether there were uncommitted changes
- the full config and seed
- the dataset path, size and SHA-256 hash, so results show exactly which data they used
- parameter count
- system info (Python, PyTorch, CUDA, GPU)
- training and validation loss at every evaluation, and after the final step
- final validation perplexity
- training throughput (tokens/sec, excluding evaluation) and peak GPU memory

Results are saved to `results/dev/` by default, which is not tracked by git. Research runs are saved with `--results-dir <directory>` so they can be committed. <br>
`results.load_runs(<directory>)` loads every run in a directory for comparison.

`python compare.py [<results_directory> ...]` prints the runs side by side (default `results/dev/`). It shows only the config settings that differ between runs, with the final losses, throughput, peak memory and commit, and warns when runs used different data, tokenizers or hardware, or had uncommitted changes. `--experiment <name>` only compares runs whose experiment name contains `<name>`.

### To benchmark a model
`python benchmark.py --config <path_to_config_file>` <br>
`python benchmark.py --model <checkpoint_path>` <br>

Measures speed and memory separately from quality, on random tokens, so it doesn't need the dataset or a trained model:
- training step (forward, backward and optimizer update) and evaluation step: time and tokens/sec at the config's `batch_size × max_seq_len`
- generation of `--num-tokens` tokens (default 200) for one sequence, after prompts of each length in `--prompt-lens` (default 1 and `max_seq_len`): total time, time to the first token, ms per token and tokens/sec
- peak GPU memory for each of these, and the memory taken by the model's parameters and buffers

Each benchmark is called `--warmup` times untimed (default 5), then timed `--repeats` times (default 20). Throughput uses the median time. Results are printed and saved to `results/dev/benchmarks/` with the same run id, commit, config and system info as training runs. `--results-dir` saves them elsewhere.

### Dataset and tokenization

Uses Karpathy's tinyshakespeare dataset for now. <br>
"Attention Is All You Need" implements byte-pair encoding (as mentioned in 5.1 Training Data and Batching), this project does the same. <br>

## Models

Configs are grouped by what they're for:
- `config/debug/`: tiny models that train in seconds on a CPU, to check the code runs
- `config/pilot/`: exploratory configs for finding a sensible setup, these change freely
- `config/research/`: the frozen baseline and the experiments compared against it. Empty until the baseline is chosen

### Bigram Language Model

A simple model that predicts the next token given only the current token. It is essentially a lookup table with dimensions based on the vocabulary size of the dataset. Each row corresponds to a token and contains scores for the next token.
[Implementation here.](models/bigram.py)

### Attention Language Model

Implemented in [models/attention.py](models/attention.py). This model uses a single layer of attention to learn, nothing else.

### Transformer Language Model
Model that implements the full transformer architecture as detailed in "Attention Is All You Need". Implemented in [models/transformer.py](models/transformer.py)

### Language Model Comparison

| Model Type     | Vocab Size | Dataset         | Batch Size | Iterations | Learning Rate | Max Sequence Length | Head Size | # Heads | # Layers | Validation Loss |
| -------------- | ---------- | --------------- | ---------- | ---------- | ------------- | --------------------| --------- | ------- | -------- | --------------- |
| Bigram         | 257        | tinyshakespeare | 32         | 25k        | 1e-3 (AdamW)  | -                   | -         | -       | -        | ~2.5            |
| Attention      | 257        | tinyshakespeare | 32         | 25k        | 1e-3 (AdamW)  | 16                  | 32        | 1       | -        | ~2.4            |
| Attention      | 257        | tinyshakespeare | 32         | 25k        | 1e-3 (AdamW)  | 16                  | 32        | 4       | -        | ~2.1            |
| Transformer    | 257        | tinyshakespeare | 32         | 10k        | 1e-3 (AdamW)  | 16                  | 32        | 4       | 4        | ~1.8            |
