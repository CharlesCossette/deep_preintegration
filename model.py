from math import floor
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch
import numpy as np
import pandas as pd
from pylie import SO3
from utils import *
from torch.utils.tensorboard import SummaryWriter


class RmiDataset(Dataset):
    def __init__(self, filename, window_size=200, stride=50):
        self._file = open(filename, "r")
        self._df = pd.read_csv(filename, sep=",")
        self._window_size = window_size
        self._stride = stride
        self._poses = None
        pass

    def __len__(self):
        return floor((self._df.shape[0] - self._window_size) / self._stride) + 1

    def __getitem__(self, idx):
        range_start = idx * self._stride
        range_stop = range_start + self._window_size

        if range_stop > (self._df.shape[0] + 1):
            raise RuntimeError("programming error")

        sample_data = self._df.iloc[range_start:range_stop, :].to_numpy()
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

        DR, DV, DC = get_gt_rmis(
            r_zw_a_i, v_zw_a_i, C_ab_i, r_zw_a_j, v_zw_a_j, C_ab_j, DT
        )
        self._poses = [(r_zw_a_i, v_zw_a_i, C_ab_i), (r_zw_a_j, v_zw_a_j, C_ab_j)]
        x = torch.from_numpy(np.vstack((t_data.T, gyro_data.T, accel_data.T)))
        x = x.to(torch.float32)

        y = torch.from_numpy(np.hstack((DC.flatten(), DV.flatten(), DR.flatten())))
        y = y.to(torch.float32)
        return x, y

    def get_item_with_poses(self, idx):
        x, y = self.__getitem__(idx)
        return x, y, self._poses


class RmiNet(torch.nn.Module):
    """
    Convolutional neural network based RMI estimator
    """

    def __init__(self, window_size=200):
        self._window_size = window_size
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
        x[:, 3:6, :] /= 9.80665
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
            y[idx, :] = get_rmis(x[idx, :, :])

        return y

def pose_loss(y, y_gt, with_info=False):
    DC = torch.reshape(y[:, 0:9], (y.shape[0], 3, 3))
    DC_gt = torch.reshape(y_gt[:, 0:9], (y_gt.shape[0], 3, 3))
    e_phi = LogSO3(torch.matmul(DC, torch.transpose(DC_gt, 1, 2)))
    e_v = y[:, 9:12] - y_gt[:, 9:12]
    e_r = y[:, 12:15] - y_gt[:, 12:15]
    e = torch.hstack((10 * e_phi, e_v, e_r))

    if torch.any(torch.isnan(e)):
        raise RuntimeError("Nan for some reason.")

    if with_info:
        info = {
            "C_loss": torch.sqrt(mse_loss(e_phi, torch.zeros(e_phi.shape))),
            "v_loss": torch.sqrt(mse_loss(e_v, torch.zeros(e_v.shape))),
            "r_loss": torch.sqrt(mse_loss(e_r, torch.zeros(e_r.shape))),
        }
        return mse_loss(e, torch.zeros(e.shape)), info

    return mse_loss(e, torch.zeros(e.shape))

def get_rmis(x):
    """ 
    Computes RMIs from accel and gyro data supplied as a big torch Tensor of
    dimension [7 x N], where N is the number of measurements.

    Zeroth row of tensor is timestamps, rows 1,2,3 are gyro, rows 4,5,6 are accel.
    """
    t = x[0, :]
    gyro = x[1:4, :]
    accel = x[4:7, :]

    DC = np.identity(3)
    DV = np.zeros((3, 1))
    DR = np.zeros((3, 1))

    for idx in range(1, x.shape[1]):
        dt = t[idx] - t[idx - 1]
        w = gyro[:, idx - 1].reshape((-1, 1))
        a = accel[:, idx - 1].reshape((-1, 1))
        DR += DV * dt + 0.5 * DC @ a * (dt ** 2)
        DV += DC @ (a * dt)
        DC = DC @ SO3.Exp(w * dt)

    temp = np.hstack((DC.flatten(), DV.flatten(), DR.flatten()))
    return torch.from_numpy(temp)

def get_gt_rmis(r_i, v_i, C_i, r_j, v_j, C_j, DT):
    g_a = np.array([0, 0, -9.80665]).reshape((-1, 1))
    DC = C_i.T @ C_j
    DV = C_i.T @ (v_j - v_i - DT * g_a)
    DR = C_i.T @ (r_j - r_i - v_i * DT - 0.5 * g_a * (DT ** 2))

    return DR, DV, DC
