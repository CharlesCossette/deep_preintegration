import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from dataset import RmiDataset
import pandas as pd
import torch


def test_length():
    dataset = RmiDataset("./data/processed/v1_01_easy.csv", window_size=1, stride=1)
    x = len(dataset)
    assert x == 28712


def test_length2():
    dataset = RmiDataset("./data/processed/v1_01_easy.csv", window_size=2, stride=1)
    x = len(dataset)
    assert x == 28711


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
    dataset = RmiDataset(filename, window_size="full", stride=1, gyro_bias=[0, 0, 0])
    l = len(dataset)
    x, y = dataset[0]
    x_test = torch.Tensor(pd.read_csv(filename).values)
    assert torch.allclose(x, x_test[:, 0:7].T)


def test_multiple_call():
    filename = "./data/processed/v1_01_easy.csv"
    dataset = RmiDataset(filename, window_size="full", stride=1)
    x, y = dataset[0]
    z, w = dataset[0]
    assert torch.all(x == z)
    assert torch.all(y == w)


def test_window_size_one():
    filename = "./data/processed/v1_01_easy.csv"
    dataset = RmiDataset(filename, window_size=1, stride=1, gyro_bias=[0, 0, 0])
    x_test = torch.Tensor(pd.read_csv(filename).values)
    l = len(dataset)
    x_list = []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        x_list.append(x)

    x_list = torch.hstack(x_list)
    assert torch.allclose(x_list, x_test[:, 0:7].T)


def test_get_time():
    filename = "./data/processed/v1_01_easy.csv"
    dataset = RmiDataset(filename, window_size=1, stride=50, gyro_bias=[0, 0, 0])
    idx = dataset.get_index_of_time(50)
    x, y = dataset[idx]
    assert torch.allclose(x[0, 0], torch.Tensor([50]))


if __name__ == "__main__":
    test_full()
