# Begin with the simplest possible model - bigram language model
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path

torch.manual_seed(1337)
block_size = 8
batch_size = 32
n_embed = 32
n_steps = 10_000


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # Need a language modeling head to go from n_embed up to vocab size
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        """Forward pass. If targets is none, just return the logits."""
        # idx and targets are both (B, T) batches of integers
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        logits = self.lm_head(token_embeddings)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # cross entropy expects a (B, C, d...)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Sample from the model."""
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def main():
    # Read in data, inspect and convert to mapping

    filepath = Path(__file__).absolute().parent / "input.txt"
    with open(filepath, "r") as infile:
        text = infile.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    STOI = {character: integer for integer, character in enumerate(chars)}
    ITOS = {integer: character for integer, character in enumerate(chars)}

    def encode(input_: str) -> "list[int]":
        return [STOI[char] for char in input_]

    def decode(input_: "list[int]") -> str:
        return "".join([ITOS[integer] for integer in input_])

    data = torch.tensor(encode(text), dtype=torch.long)

    # Convert the data into a train/test split
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]

    def get_batch(split: str):
        """Generate a mini-batch of training data."""
        if split not in ["train", "test"]:
            raise ValueError("`split` must be one of `train` or `test`.")

        data = train_data if split == "train" else test_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])  # noqa
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # noqa
        return x, y

    xb, yb = get_batch("train")
    model = BigramLanguageModel(vocab_size)
    logits, loss = model(idx=xb, targets=yb)

    # torch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for steps in tqdm(range(n_steps)):
        xb, yb = get_batch("train")

        logits, loss = model(idx=xb, targets=yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def print_sample(bigram_model, max_new_tokens: int = 100):
        idx = torch.zeros((1, 1), dtype=torch.long)
        print(decode(bigram_model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()))

    print_sample(model, max_new_tokens=300)


if __name__ == "__main__":
    main()
