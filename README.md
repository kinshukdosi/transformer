# Transformer Systems Lab

A decoder-only Transformer language model written from scratch in PyTorch, being developed into a measured, reproducible study of how architecture and inference choices affect quality, memory, throughput and latency on a single consumer GPU (RTX 3060, 12 GB).

The emphasis is on understanding and documenting every step: each component is explained in the code, tested against a reference or an invariant, and every training run records enough to be reproduced and compared.

## Status

- [x] Bigram, attention-only and full Transformer language models
- [x] Correctness tests: MHA/GQA/MQA against a reference, causal masking, block structure, gradients, tokenizers, checkpoints, seeding and exact resume
- [x] Structured run results, run comparison and a speed/memory benchmark runner
- [x] Reference check: matches nanoGPT's Tiny Shakespeare result ([below](#reference-check-nanogpt-shakespeare_char))
- [ ] Model-size and tokenizer measurements
- [ ] Research dataset selection and a frozen baseline
- [ ] Architecture experiments (positional encoding, normalization, feed-forward, KV heads) and a KV cache

## Results

### Reference check: nanoGPT shakespeare_char

To check the implementation against a known result, [`config/pilot/nanogpt_reference.yaml`](config/pilot/nanogpt_reference.yaml) uses the same model shape and training settings as [nanoGPT](https://github.com/karpathy/nanoGPT)'s character-level Tiny Shakespeare config: 6 layers, 6 heads, 384 dimensions, 256-character context, batch size 64, 5000 steps, learning rate 1e-3 with 100 warmup steps and cosine decay to 1e-4, dropout 0.2, weight decay 0.1, gradient clipping at 1.0, seed 1337.

|                                | This repo                    | nanoGPT |
| ------------------------------ | ---------------------------- | ------- |
| Best validation loss           | **1.4684** (step 1750)       | 1.4697  |
| Final validation loss (step 5000) | 1.6402                    | -       |
| Final training loss (step 5000)   | 0.6930                    | -       |
| Parameters                     | 10,786,625                   | -       |
| Training throughput            | 67,260 tokens/sec            | -       |
| Training time                  | 20.3 min (excluding evaluation) | -    |
| Peak GPU memory                | 3,482 MB                     | -       |

RTX 3060 12 GB, fp32, PyTorch 2.11.0, CUDA 13.0, commit `b02a423`. Losses are per character. [Full results, including every evaluation.](results/nanogpt-reference/20261001-112118-pilot-nanogpt-reference-a65ba1.json)

The best validation loss matches nanoGPT's reported best. After step 1750 the model overfits: training loss keeps falling while validation loss rises from 1.47 to 1.64. 5000 steps of 64 × 256 characters is 82M characters, about 80 passes over the 1M-character training split. Tiny Shakespeare is therefore used here for development and checking correctness, not as the dataset for the research results.

The models aren't identical. nanoGPT uses GELU instead of ReLU, ties the input embedding and output projection weights, and has no biases.

## Architecture

Three language models of increasing complexity, all predicting the next token:

- **Bigram** ([`models/bigram.py`](models/bigram.py)): a `vocab_size × vocab_size` lookup table of next-token scores, given only the current token.
- **Attention** ([`models/attention.py`](models/attention.py)): token and position embeddings followed by one layer of causal self-attention. No feed-forward network or normalization.
- **Transformer** ([`models/transformer.py`](models/transformer.py)): the full architecture, described below.

| Component | Choice |
| --- | --- |
| Positional encoding | Learned absolute position embeddings, added to the token embeddings |
| Block structure | Pre-LN: `x + Attention(LayerNorm(x))`, then `x + FeedForward(LayerNorm(x))`, repeated `n_layers` times |
| Final normalization | LayerNorm before the output projection, because Pre-LN never normalizes the residual stream itself |
| Attention | Causal multi-head attention, scaled by `1/sqrt(head_size)`. MHA, GQA and MQA through `group_size`, the number of query heads sharing one key/value head |
| Feed-forward | `Linear(n_embd, 4·n_embd)` → ReLU → `Linear(4·n_embd, n_embd)` |
| Dropout | On the embeddings, the attention weights and the output of both sub-layers |
| Initialization | Weights from N(0, 0.02) and zero biases, like GPT-2. Projections that write into the residual stream are scaled down by `1/sqrt(2·n_layers)` |
| Output | Separate linear projection to `vocab_size`, not tied to the input embedding |

`group_size` sets the attention variant: `1` is multi-head attention (MHA), `num_heads` is multi-query attention (MQA, one key/value head), anything in between is grouped-query attention (GQA, `num_heads / group_size` key/value heads).

Training uses AdamW with weight decay on weight matrices and embeddings only, optional linear warmup and cosine learning-rate decay, and optional gradient clipping.

## Quick start

Requires Python 3.12.

```bash
pip install -r requirements.txt
pytest                                                      # run the tests

python train.py --config config/debug/transformer.yaml      # seconds on a CPU
python train.py --config config/pilot/nanogpt_reference.yaml --save model.pt
python generate.py --model model.pt --prompt "ROMEO:"
python benchmark.py --config config/pilot/nanogpt_reference.yaml
python compare.py                                           # compare runs in results/dev/
```

## Reproducibility

- **Seeded.** Training is seeded with the config's `seed`, so the same config trains the same model on the same setup.
- **Exact resume.** Checkpoints include the optimizer, the step and the random number generator state, so a resumed run gives exactly the same result as an uninterrupted one.
- **Comparable evaluation.** Losses are measured on the same fixed windows of each split every time, without using the random number generator, so evaluation settings don't change training.
- **No leakage.** The tokenizer is trained on the training split only.
- **Recorded.** Every run saves a JSON file with the git commit (and whether there were uncommitted changes), the full config, a hash of the dataset, software and hardware versions, every evaluation, throughput and peak memory.

Development runs are saved to `results/dev/`, which isn't tracked by git. Results reported in this README are committed under `results/`, and each links to its file.

## Project structure

```
models/          bigram, attention-only and Transformer language models
config/
  debug/         tiny models that train in seconds on a CPU, to check the code runs
  pilot/         exploratory configs for finding a sensible setup, these change freely
  research/      the frozen baseline and the experiments compared against it (empty until the baseline is chosen)
data/            Tiny Shakespeare
results/         committed run results. results/dev/ holds untracked development runs
tests/           pytest suite
train.py         training loop, evaluation, checkpoints and run results
generate.py      text generation from a checkpoint
benchmark.py     training, evaluation and generation speed and memory
compare.py       side-by-side comparison of run results
config.py        YAML configs → dataclasses, and model/tokenizer factories
tokenizer.py     character-level and byte-pair encoding tokenizers
data_loader.py   train/validation split, random training batches and fixed evaluation windows
results.py       run result records
```

## References

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019): Pre-LN with a final LayerNorm, weight initialization
- Ba et al., ["Layer Normalization"](https://arxiv.org/abs/1607.06450) (2016)
- Srivastava et al., ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) (2014)
- Ainslie et al., ["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/abs/2305.13245) (2023)
- Andrej Karpathy, ["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY), [nanoGPT](https://github.com/karpathy/nanoGPT) and [minbpe](https://github.com/karpathy/minbpe)

---

## Reference

### Training

`python train.py --config <config.yaml>`

- `--iterations <n>` overrides the number of iterations in the config
- `--save [<path.pt>]` saves a checkpoint at the end of training (default `checkpoint.pt`)
- `--load [<path.pt>]` resumes from a checkpoint (default `checkpoint.pt`). `iterations` is the total number of steps, so `--load model.pt --iterations 20000` continues up to step 20000. Ignored if `--config` is also passed
- `--results-dir <directory>` saves the run results somewhere other than `results/dev/`

Checkpoints contain the config, model and optimizer state, the trained tokenizer, the step reached and the random number generator state, so generating text doesn't need the dataset.

### Config

Configs are YAML files. See [`config/debug/transformer.yaml`](config/debug/transformer.yaml) for a complete example.

| Key | Meaning |
| --- | --- |
| `model_type` | `bigram`, `attention` or `transformer` |
| `tokenizer`, `vocab_size` | `simple` (character-level) or `bpe` (byte-pair encoding, `vocab_size` > 256) |
| `data_path`, `train_split` | Text file, and the fraction of it used for training. The rest is the validation split |
| `max_seq_len` | Context length: the most tokens the model can attend to |
| `batch_size`, `iterations` | Sequences per step, and the total number of training steps |
| `n_embd`, `num_heads`, `group_size`, `n_layers`, `dropout` | Model shape (attention and Transformer models). `group_size` is the number of query heads per key/value head |
| `optimizer`, `lr`, `beta1`, `beta2`, `eps` | AdamW. Defaults are PyTorch's |
| `weight_decay` | Applied to weight matrices and embeddings only, not to biases or LayerNorm parameters (default 0.01) |
| `warmup_iters` | The learning rate increases linearly from close to 0 up to `lr` over this many steps (default 0) |
| `lr_decay_iters`, `min_lr` | After the warmup, the learning rate follows a cosine curve from `lr` to `min_lr` at step `lr_decay_iters`, then stays at `min_lr`. Set both or neither (default: no decay). The schedule depends only on the step, so stopping early or resuming for longer doesn't change it |
| `grad_clip` | If the combined norm of all gradients is above this, they are scaled down to it (default: no clipping) |
| `eval_interval`, `eval_iterations` | Evaluate every `eval_interval` steps, on up to `eval_iterations × batch_size` windows spread evenly across each split |
| `seed` | Random seed (default 100) |
| `experiment` | Name recorded in the results and used in the results file name |

### Generation

`python generate.py --model <checkpoint.pt> --prompt "ROMEO:"`

- `--num-tokens <n>` sets how many tokens to generate (default 500)
- `--greedy` always picks the most likely next token instead of sampling, so the output is the same every time. It tends to get stuck repeating itself, but it is useful for checking that two ways of generating give exactly the same tokens

### Run results

Every training run saves a JSON file with:
- run id, experiment name, git commit, and whether tracked files had uncommitted changes
- the full config and seed
- the dataset path, size and SHA-256 hash
- parameter count
- system info (Python, PyTorch, CUDA, GPU)
- training and validation loss and the learning rate at every evaluation, and after the final step
- final validation perplexity (per token, so only comparable between runs with the same tokenizer)
- training throughput (tokens/sec, excluding evaluation) and peak GPU memory

`results.load_runs(<directory>)` loads every run in a directory.

`python compare.py [<results_directory> ...]` prints runs side by side (default `results/dev/`). It shows only the config settings that differ between runs, with the final losses, throughput, peak memory and commit, and warns when runs used different data, tokenizers or hardware, or had uncommitted changes. `--experiment <name>` only compares runs whose experiment name contains `<name>`.

### Benchmarking

`python benchmark.py --config <config.yaml>` or `python benchmark.py --model <checkpoint.pt>`

Measures speed and memory separately from quality, on random tokens, so it doesn't need the dataset or a trained model:
- training step (forward, backward, gradient clipping and optimizer update) and evaluation step: time and tokens/sec at the config's `batch_size × max_seq_len`
- generation of `--num-tokens` tokens (default 200) for one sequence, after prompts of each length in `--prompt-lens` (default 1 and `max_seq_len`): total time, time to the first token, ms per token and tokens/sec
- peak GPU memory for each of these, and the memory taken by the model's parameters and buffers

Each benchmark is called `--warmup` times untimed (default 5), then timed `--repeats` times (default 20). Throughput uses the median time. Results are printed and saved to `results/dev/benchmarks/` with the same run id, commit, config and system info as training runs. `--results-dir` saves them elsewhere.
