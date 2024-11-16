# Neural Networks Zero to One

A reproduction of the excellent work of [Andrej Karpathy](https://karpathy.ai) and his [Neural Networks Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) Youtube series.

This repository is a learning exercise that serves two purposes:

1. Familiarization with and learning about modern natural language processing.
2. Development and deploying of neural networks using the [Modular MAX Engine](https://www.modular.com/max).

The code here will be developed in three phases:

1. Working through all of the python based examples in Karpathy's Youtube playlist.
2. Conversion of the python based components to [Mojo](https://www.modular.com/mojo) code for interop with `MAX`.
3. Containerization of the transformer model for deployment as a self-hosted application.

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
