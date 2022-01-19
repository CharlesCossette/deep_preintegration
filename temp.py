import torch
from torch.utils.data import ConcatDataset, Subset
from model import DeltaTransRmiNet
from dataset import RmiDataset

N = 1  # Window size
stride = 1

torch.set_default_dtype(torch.float64)
with_model = False
raw_datasets = [
    RmiDataset("./data/processed/v1_03_difficult.csv", N, stride, with_model),
    RmiDataset("./data/processed/v2_01_easy.csv", N, stride, with_model),
    RmiDataset("./data/processed/v2_02_medium.csv", N, stride, with_model),
    RmiDataset("./data/processed/v2_03_difficult.csv", N, stride, with_model),
    RmiDataset("./data/processed/v1_02_medium.csv", N, stride, with_model),
]

trainset_list = []
validset_list = []
for dataset in raw_datasets:
    idx = dataset.get_index_of_time(50)
    trainset_list.append(Subset(dataset, list(range(idx))))
    validset_list.append(Subset(dataset, list(range(idx, len(dataset)))))

trainset = ConcatDataset(trainset_list)
validset = ConcatDataset(validset_list)

x_list = []
for x, y in trainset:
    x_list.append(x)

x_all = torch.hstack(x_list)
mean_imu = torch.mean(x_all[1:, :], dim=1)
std_imu = torch.std(x_all[1:, :], dim=1, unbiased=True)

print(mean_imu)
print(std_imu)
