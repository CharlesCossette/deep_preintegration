from model import RmiDataset


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


if __name__ == "__main__":
    test_length2()
