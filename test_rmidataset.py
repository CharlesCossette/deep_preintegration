from model import RmiDataset
import pandas as pd
import torch


def test_length():
    dataset = RmiDataset("./data/processed/v1_01_easy.csv", window_size=1, stride=1)
    x = len(dataset)
    assert x == 29120


def test_length2():
    dataset = RmiDataset("./data/processed/v1_01_easy.csv", window_size=2, stride=1)
    x = len(dataset)
    assert x == 29119


def test_window_size():
    dataset = RmiDataset("./data/processed/v1_01_easy.csv", window_size=5, stride=1)
    for i in range(len(dataset)):
        dataset[i]


def test_stride():
    dataset = RmiDataset("./data/processed/v1_01_easy.csv", window_size=1, stride=15)
    for i in range(len(dataset)):
        dataset[i]


def test_stride_window():
    dataset = RmiDataset("./data/processed/v1_01_easy.csv", window_size=6, stride=15)
    for i in range(len(dataset)):
        dataset[i]


def test_full():
    filename = "./data/processed/v1_01_easy.csv"
    dataset = RmiDataset(filename, window_size="full", stride=1)
    l = len(dataset)
    x, y = dataset[0]
    x_test = torch.Tensor(pd.read_csv(filename).values)
    assert torch.allclose(x, x_test[:, 0:7].T)


if __name__ == "__main__":
    test_full()
