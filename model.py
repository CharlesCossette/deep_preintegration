from math import floor
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch
import numpy as np
import pandas as pd
from pylie import SO3
from torch.utils.tensorboard import SummaryWriter

# TODO: check gravity direction


class RmiDataset(Dataset):
    def __init__(self, filename, window_size=200, stride=50):
        self._file = open(filename, "r")
        self._df = pd.read_csv(filename, sep=",", header=None)
        self._window_size = window_size
        self._stride = stride
        pass

    def __len__(self):
        return floor((self._df.shape[0] - self._window_size) / self._stride) + 1
        
    def __getitem__(self, idx):
        range_start = idx * self._stride
        range_stop = range_start + self._window_size

        if range_stop > (self._df.shape[0] + 1):
            raise RuntimeError("programming error")

        sample_data = self._df[range_start:range_stop].to_numpy()
        t_data = sample_data[:, 0]
        gyro_data = sample_data[:, 1:4]
        accel_data = sample_data[:, 4:7]

        DT = t_data[-1] - t_data[0]
        g_a = np.array([0, 0, -9.80665]).reshape((-1, 1))
        r_zw_a_i = sample_data[0, 7:10].reshape((-1, 1))
        r_zw_a_j = sample_data[-1, 7:10].reshape((-1, 1))
        v_zw_a_i = sample_data[0, 10:13].reshape((-1, 1))
        v_zw_a_j = sample_data[-1, 10:13].reshape((-1, 1))
        C_ab_i = sample_data[0, 13:]
        C_ab_j = sample_data[-1, 13:]
        C_ab_i = C_ab_i.reshape((3, 3))
        C_ab_j = C_ab_j.reshape((3, 3))

        DC = C_ab_i.T @ C_ab_j
        DV = C_ab_i.T @ (v_zw_a_j - v_zw_a_i - DT * g_a)
        DR = C_ab_i.T @ (
            r_zw_a_j - r_zw_a_i - v_zw_a_i * DT - 0.5 * g_a * DT ** 2
        )

        x = torch.from_numpy(np.vstack((t_data.T, gyro_data.T, accel_data.T)))
        x = x.to(torch.float32)

        y = torch.from_numpy(
            np.hstack((DC.flatten(), DV.flatten(), DR.flatten()))
        )
        y = y.to(torch.float32)
        return x, y


class RmiNet(torch.nn.Module):
    """
    Convolutional neural network based RMI estimator
    """
    def __init__(self, window_size=200):
        super(RmiNet, self).__init__()

        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(6, 6, 5, padding=4),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv1d(6, 1, 5, padding=4, dilation=3),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Flatten(),
        )
        self.linear_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(30),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(30, 15),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.linear_layer(x)

        # Normalize the rotation matrix to make it a valid element of SO(3)
        R = torch.reshape(x[:, 0:9], (x.shape[0], 3, 3))
        U, _, VT = torch.linalg.svd(R)
        S = torch.eye(3).reshape((1, 3, 3)).repeat(x.shape[0], 1, 1)
        S[:, 2, 2] = torch.det(torch.matmul(U, VT))
        R_norm = torch.matmul(U, torch.matmul(S, VT))
        R_flat = torch.reshape(R_norm, (x.shape[0], 9))

        # TODO: double check to see if the below operation works as intended
        return torch.cat((R_flat, x[:, 9:]), 1)


class RmiModel(torch.nn.Module):
    def __init__(self, window_size=200):
        super(RmiModel, self).__init__()

    def forward(self, x):
        # shape[0] is the batch size
        x = x.detach().to("cpu").numpy()
        y = torch.zeros((x.shape[0], 15))
        for idx in range(x.shape[0]):
            y[idx, :] = self._get_rmis(x[idx, :, :])

        return y

    def _get_rmis(self, x):
        t = x[0, :]
        gyro = x[1:4, :]
        accel = x[4:7, :] * 9.80665

        DC = np.identity(3)
        DV = np.zeros((3, 1))
        DR = np.zeros((3, 1))

        for idx in range(1, x.shape[1]):
            dt = t[idx] - t[idx - 1]
            w = gyro[:, idx - 1].reshape((-1, 1))
            a = accel[:, idx - 1].reshape((-1, 1))
            DR += DV * dt + 0.5 * DC @ a * dt ** 2
            DV += DC @ a * dt
            DC = DC @ SO3.Exp(w * dt)

        temp = np.hstack((DC.flatten(), DV.flatten(), DR.flatten()))
        return torch.from_numpy(temp)


    # num_samples = len(alldata)
    # num_training = round(0.7 * num_samples)
    # num_valid = round(0.2 * num_samples)
    # num_test = num_samples - num_training - num_valid
    # datasets = random_split(
    # alldata,
    # [num_training, num_valid, num_test],
    # generator=torch.Generator().manual_seed(42),
    # )
    # trainset = datasets[0]
    # validset = datasets[1]



  