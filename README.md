# Neural Networks Zero to One

A reproduction of the excellent work of [Andrej Karpathy](https://karpathy.ai) and his [Neural Networks Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) Youtube series.

This repository is a learning exercise that serves two purposes:

1. Familiarization with and learning about modern natural language processing.
2. Development and deploying of neural networks using the [Modular MAX Engine](https://www.modular.com/max).

## Environment Setup

Setup for this development environment requires the [magic](https://docs.modular.com/magic/) package manager.

```bash
magic add --manifest-path mojoproject.toml
```

## Data

The input data for `makemore` can be obtained via:

```bash
wget https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
```

The input data for `nanoGPT` can be obtained via:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Development Plan

1. Complete the neural networks Youtube series as a collection of Jupyter notebooks.  
2. Re-implement logic from notebooks as a small Python module.  
3. Re-implement Python code in Mojo.  
4. Create a local model deployment using MAX engine.  
