import json
import torch
from tqdm import tqdm
from pathlib import Path

from self_attention import SelfAttentionModel


def encode(stoi: "dict[str: int]", input_: str) -> "list[int]":
    return [stoi[char] for char in input_]


def decode(itos: "dict[int: str]", input_: "list[int]") -> str:
    return "".join([itos[integer] for integer in input_])


def get_batch(data: "torch.tensor", config: "dict[str: int | float | bool]"):
    """Generate a mini-batch of data."""
    block_size = config.get("block_size")
    batch_size = config.get("batch_size")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])  # noqa
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # noqa
    return x, y


def print_sample(model, itos: "dict[int: str]", max_new_tokens: int = 100):
    idx = torch.zeros((1, 1), dtype=torch.long)
    print(decode(itos, model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()))


@torch.no_grad()
def estimate_loss(
    model, *, train_data: "torch.tensor", test_data: "torch.tensor", config: "dict[str: int | float | bool]"
):
    out = {}
    model.eval()
    eval_iters = config.get("eval_iters")

    for split, data in zip(["train", "test"], [train_data, test_data]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()

    return out


def main():
    working_directory = Path(__file__).absolute().parent

    filepath = working_directory / "input.txt"
    with open(filepath, "r") as infile:
        text = infile.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    STOI = {character: integer for integer, character in enumerate(chars)}
    ITOS = {integer: character for integer, character in enumerate(chars)}

    data = torch.tensor(encode(STOI, text), dtype=torch.long)

    # Convert the data into a train/test split
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]

    filepath = working_directory / "train_config.json"
    with open(filepath, "r") as infile:
        config = json.load(infile)

    model = SelfAttentionModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    losses = {}
    for steps in tqdm(range(config["max_iters"])):
        xb, yb = get_batch(train_data, config=config)
        logits, loss = model(idx=xb, targets=yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if steps % 100 == 0 and config["print_losses"]:
            loss_estim = estimate_loss(model, train_data=train_data, test_data=test_data, config=config)
            losses[steps] = loss_estim

    print_sample(model, itos=ITOS, max_new_tokens=config["max_new_tokens"])

    if config["print_losses"]:
        for loss_estim in losses.values():
            print(f"Training loss: {loss_estim['train']}, Test loss: {loss_estim['test']}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    outfile = working_directory / config["model_checkpoint_name"]
    torch.save(checkpoint, outfile)


if __name__ == "__main__":
    main()
