import os
import gzip
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import matplotlib.pyplot as plt

DATA_DIR = "./download/data"


def load(filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'rb') as f:
        content = f.read()
    return np.frombuffer(gzip.decompress(content), dtype=np.uint8).copy()


def train(model, X_train, Y_train):
    epochs = 1000
    batch_size = 256
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.01)
    losses = []
    accuracies = []
    for i in (trng := trange(epochs)):
        rand_samp = np.random.randint(0, X_train.shape[0], size=batch_size)
        X_samp = torch.tensor(X_train[rand_samp].reshape(-1, 28 * 28)).float()
        Y_samp = torch.tensor(Y_train[rand_samp]).long()
        optimizer.zero_grad()
        out = model(X_samp)
        cat = torch.argmax(out, dim=1)
        accuracy_t = (cat == Y_samp).float().mean()
        loss_t = loss_fn(out, Y_samp)
        loss_t.backward()
        optimizer.step()
        loss = loss_t.item()
        accuracy = accuracy_t.item()
        losses.append(loss)
        accuracies.append(accuracy)
        trng.set_description(f"loss: {loss:.2f} / accuracy: {accuracy:.2f} / progress")
    # plt.ylim(-0.1, 1.1)
    # plt.plot(losses)
    # plt.plot(accuracies)
    # plt.show()


def main():
    from torchnet import MyNet
    from npnet import NpNet

    model = MyNet()
    X_train = load("x-train.gz")[0x10:].reshape(-1, 28*28)
    Y_train = load("y-train.gz")[0x8:]
    X_test = load("x-test.gz")[0x10:].reshape(-1, 28*28)
    Y_test = load("y-test.gz")[0x8:]

    train(model, X_train, Y_train)

    # test evaluation:
    Y_pred = torch.argmax(
        model(torch.tensor(X_test.reshape(-1, 28 * 28)).float()),
        dim=1
    ).numpy()
    test1_acc = np.equal(Y_test, Y_pred).mean()
    print(f"test1 accuracy: {test1_acc}")

    model2 = NpNet(
        model.l1.weight.detach().numpy().transpose(),
        model.l2.weight.detach().numpy().transpose()
    )
    Y_preds2 = model2.predict(X_test)
    test2_acc = np.equal(Y_test, np.argmax(Y_preds2, axis=1)).mean()
    print(f"test2 accuracy: {test2_acc}")

    # print scores:
    #print(np.array([Y_preds2[i, Y_test[i]] for i in range(Y_preds2.shape[0])]))

    # samp = range(0, 10)
    losses = model2.loss(Y_preds2, Y_test)
    max_loss = np.argmax(losses)
    plt.imshow(X_test[max_loss].reshape(28, 28))
    print("worst image: ", Y_test[max_loss])
    plt.show()


if __name__ == '__main__':
    main()
