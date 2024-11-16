import matplotlib.pyplot as plt
from micrograd.nn import MLP

learning_rate = 0.05
hidden = [4, 4, 4, 1]
mlp = MLP(3, hidden)

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

loss_data = list()
for epoch in range(1000):

    # forward pass
    ypred = [mlp(x) for x in xs]
    loss = sum((y_out - y_gt) ** 2 for y_out, y_gt in zip(ypred, ys))

    # zero gradients
    for p in mlp.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in mlp.parameters():
        p.data += -learning_rate * p.grad

    loss_data.append((epoch, loss.data))

plt.plot(*zip(*loss_data))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss per epoch")
plt.savefig("loss.png")
plt.close()
