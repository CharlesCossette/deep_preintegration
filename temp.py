import torch
from torch.utils.data import ConcatDataset, Subset
from model import RmiModel
from dataset import RmiDataset, DeltaRmiDataset
import matplotlib.pyplot as plt

N = 500  # Window size
output_window = 500
stride = 10
calib_file = "./results/best_calibration_saved.pt"

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
    idx = dataset.get_index_of_time(100)
    trainset_list.append(Subset(dataset, list(range(idx))))
    validset_list.append(Subset(dataset, list(range(idx, len(dataset)))))

trainset = ConcatDataset(trainset_list)
validset = ConcatDataset(validset_list)

# x_list = []
# for x, y in trainset:
#     x_list.append(x)

# x_all = torch.hstack(x_list)
# mean_imu = torch.mean(x_all[1:, :], dim=1)
# std_imu = torch.std(x_all[1:, :], dim=1, unbiased=True)

# print(mean_imu)
# print(std_imu)

calib_results = torch.load(calib_file, map_location="cpu")
calibrator = RmiModel(output_window = output_window)
calibrator.load_state_dict(calib_results)

delta_trainset = DeltaRmiDataset(trainset, calibrator)
delta_validset = DeltaRmiDataset(validset, calibrator)


mean_gyro = torch.mean(delta_trainset.x[:,1:4,:], dim=2)
mean_accel = torch.mean(delta_trainset.x[:,4:7,:], dim=2)
meas = torch.hstack((mean_gyro, mean_accel))
e = delta_trainset.e
fig, axs = plt.subplots(6, 9)
idx = 0
for i in range(6):
    for j in range(9):
        axs[i][j].scatter(e[:,j], meas[:,i], 1)

plt.show()