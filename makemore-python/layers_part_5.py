# Neural net components for part of the series - Wavenet
import torch


class Linear:
    """A basic linear unit representing the operation y = xA^T + b with optional bias."""

    def __init__(
        self,
        fan_in: float,
        fan_out: float,
        bias: bool = True,
    ):
        kaiming_init = fan_in**0.5
        self.weight = torch.randn((fan_in, fan_out)) / kaiming_init
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x: "torch.tensor") -> "torch.tensor":
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    @property
    def parameters(self) -> "list[torch.tensor]":
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """Basic 1D batchnorm implementation that maintains stats during training."""

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)  # batch norm gain
        self.beta = torch.zeros(dim)  # batch norm shift

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x: "torch.tensor") -> "torch.tensor":
        if self.training:
            if x.ndim == 2:
                reduce_dim = 0
            elif x.ndim == 3:
                reduce_dim = (0, 1)

            x_mean = x.mean(reduce_dim, keepdim=True)
            x_var = x.var(reduce_dim, keepdim=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta

        # Track the batch-norm statistics as training progresses
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var
        return self.out

    @property
    def parameters(self) -> "list[torch.tensor]":
        """Returns the gain and the norm shift as a list of tensors."""
        return [self.gamma, self.beta]


class Tanh:
    """Wrapper around the torch.tanh activation function."""

    def __call__(self, x: "torch.tensor"):
        self.out = torch.tanh(x)
        return self.out

    @property
    def parameters(self) -> "list":
        return []


class Embedding:

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, ix):
        self.out = self.weight[ix]
        return self.out

    @property
    def parameters(self) -> "list":
        return [self.weight]


class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    @property
    def parameters(self) -> "list":
        return []


class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    @property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]
