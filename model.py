from math import floor
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch
import pandas as pd
from pylie.torch import SO3
from utils import *
from torch.utils.tensorboard import SummaryWriter


class RmiDataset(Dataset):
    """
    Generic dataset object that extracts RMIs from the IMU measurements over a
    specified window size. Requires the data to be stored in a csv file where
    the columns are:

    | |Gyro    |Accel   |Pos     |Vel     |Rotation (row-major format)
     t,wz,wy,wz,ax,ay,az,rx,ry,rz,vx,vy,vz,c11,c12,c12,c21,c22,c23,c31,c32,c33

    PARAMETERS:
    -----------
    filename: string
        path to csv
    window_size: int or "full"
        Number of IMU measurements involved in the RMIs. If "full" then
        the window size spans the entire dataset.
    stride: int
        Step interval when looping through the dataset

    """

    def __init__(
        self,
        filename,
        window_size=200,
        stride=1,
        with_model=False,
    ):
        self._file = open(filename, "r")
        self._df = pd.read_csv(filename, sep=",")

        if window_size == "full":
            window_size = len(self._df)

        self._window_size = window_size
        self._stride = stride
        self._with_model = with_model
        self._poses = None
        self._quickloader = [None] * self.__len__()
        pass

    def __len__(self):
        return floor((self._df.shape[0] - self._window_size) / self._stride) + 1

    def __getitem__(self, idx):
        # Get from quickloader if available, otherwise compute from scratch.
        out = self._quickloader[idx]
        if out is None:
            out = self._compute_sample(idx)
        return out

    def _compute_sample(self, idx):
        range_start = idx * self._stride
        range_stop = range_start + self._window_size

        if range_stop > (self._df.shape[0] + 1):
            raise RuntimeError("programming error")

        # Convert to Torch tensor.
        sample_data = torch.Tensor(self._df.iloc[range_start:range_stop, :].values)

        # Get network input: gyro and accel data.
        t_data = sample_data[:, 0]
        gyro_data = sample_data[:, 1:4]
        accel_data = sample_data[:, 4:7]
        x = torch.vstack((t_data.T, gyro_data.T, accel_data.T))

        t_i = t_data[0]
        t_j = t_data[-1]
        DT = t_j - t_i
        r_i = sample_data[0, 7:10].view((-1, 1))
        r_j = sample_data[-1, 7:10].view((-1, 1))
        v_i = sample_data[0, 10:13].view((-1, 1))
        v_j = sample_data[-1, 10:13].view((-1, 1))
        C_i = sample_data[0, 13:]
        C_j = sample_data[-1, 13:]
        C_i = C_i.view((3, 3))
        C_j = C_j.view((3, 3))

        # Get ground truth RMIs from ground truth pose information
        DR_gt, DV_gt, DC_gt = get_gt_rmis(r_i, v_i, C_i, r_j, v_j, C_j, DT)
        y = flatten_pose(DR_gt, DV_gt, DC_gt)

        # Get RMIs from analytical model
        if self._with_model:
            y = torch.hstack((y, get_rmis(x)))

        # Store absolute poses for convenience
        self._poses = [
            flatten_pose(r_i, v_i, C_i),
            flatten_pose(r_j, v_j, C_j),
        ]

        # Store this result in quickloader for easy access when this index is
        # called again.
        self._quickloader[idx] = (x, y)
        return x, y

    def get_item_with_poses(self, idx):
        """
        Provides a sample with an additional third output: the absolute poses
        at the beginning and end of the window.
        """
        x, y = self.__getitem__(idx)
        return x, y, self._poses
    
    def reset(self):
        """
        Clears the internal quickloader so that all samples must be computed
        from scratch.
        """
        self._quickloader = [None] * self.__len__()



class RmiNet(torch.nn.Module):
    """
    Convolutional neural network based DIRECT RMI estimator.
    """

    def __init__(self, window_size=200):
        self._window_size = window_size
        super(RmiNet, self).__init__()

        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(6, 32, 5, padding=4),
            # torch.nn.LazyBatchNorm1d(),
            torch.nn.GELU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Conv1d(32, 32, 5, padding=4, dilation=3),
            # torch.nn.LazyBatchNorm1d(),
            torch.nn.GELU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Conv1d(32, 32, 5, padding=4, dilation=3),
            # torch.nn.LazyBatchNorm1d(),
            torch.nn.GELU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Conv1d(32, 1, 5, padding=4, dilation=3),
            # torch.nn.LazyBatchNorm1d(),
            torch.nn.GELU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Flatten(),
        )
        self.linear_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            # torch.nn.LazyBatchNorm1d(),
            torch.nn.GELU(),
            # torch.nn.Dropout(p=0.3),
            torch.nn.Linear(50, 50),
            # torch.nn.LazyBatchNorm1d(),
            torch.nn.GELU(),
            # torch.nn.Dropout(p=0.3),
            torch.nn.Linear(50, 15),
        )

    def forward(self, x: torch.Tensor):
        x = x.detach().clone()
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

        return torch.cat((R_flat, x[:, 9:]), 1)





class RmiModel(torch.nn.Module):
    """
    A pytorch module implementation of the classic IMU dead-reckoning RMIs
    from Forster et. al. (2017).
    """
    def __init__(self, window_size=200):
        super(RmiModel, self).__init__()

    def forward(self, x):
        # shape[0] is the batch size
        y = torch.zeros((x.shape[0], 15))
        for idx in range(x.shape[0]):
            y[idx, :] = get_rmis(x[idx, :, :])

        return y


def pose_loss(y, y_gt, with_info=False):
    DC = torch.reshape(y[:, 0:9], (y.shape[0], 3, 3))
    DC_gt = torch.reshape(y_gt[:, 0:9], (y_gt.shape[0], 3, 3))
    e_phi = SO3.Log(torch.matmul(torch.transpose(DC, 1, 2), DC_gt))
    e_v = y[:, 9:12] - y_gt[:, 9:12]
    e_r = y[:, 12:15] - y_gt[:, 12:15]
    e = torch.hstack((20 * e_phi, e_v, e_r))

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

    DC = torch.eye(3)
    DV = torch.zeros((3, 1))
    DR = torch.zeros((3, 1))

    for idx in range(1, x.shape[1]):
        dt = t[idx] - t[idx - 1]
        w = gyro[:, idx - 1].reshape((-1, 1))
        a = accel[:, idx - 1].reshape((-1, 1))
        DR += DV * dt + 0.5 * DC @ a * (dt ** 2)
        DV += torch.matmul(DC, a) * dt
        DC = torch.matmul(DC, SO3.Exp(w * dt).squeeze())

    return flatten_pose(DR, DV, DC)


def get_gt_rmis(r_i, v_i, C_i, r_j, v_j, C_j, DT):
    g_a = torch.Tensor([0, 0, -9.80665]).view((-1, 1))
    DC = C_i.T @ C_j
    DV = C_i.T @ (v_j - v_i - DT * g_a)
    DR = C_i.T @ (r_j - r_i - v_i * DT - 0.5 * g_a * (DT ** 2))

    return DR, DV, DC
