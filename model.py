import torch
from pylie.torch import SO3
from utils import flatten_pose
import numpy as np




class BaseNet(torch.nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # for normalizing inputs
        self.mean_x = torch.Tensor(
            [0.0458, 0.0114, -0.0163, 9.2852, 0.0166, -3.2451]
        ).view((-1, 1))
        self.std_x = torch.Tensor(
            [0.4135, 0.3522, 0.2884, 1.4742, 0.6551, 0.8713]
        ).view((-1, 1))
        self.mean_x = torch.nn.Parameter(self.mean_x, requires_grad=False)
        self.std_x = torch.nn.Parameter(self.std_x, requires_grad=False)
        self.I = torch.nn.Parameter(torch.eye(6), requires_grad=False)

    def norm(self, x):
        return (x - self.mean_x) / self.std_x

    def set_normalized_factors(self, mean_x, std_x):
        self.mean_u = torch.nn.Parameter(mean_x, requires_grad=False)
        self.std_u = torch.nn.Parameter(std_x, requires_grad=False)


class RmiNet(torch.nn.Module):
    """
    Convolutional neural network based DIRECT RMI estimator.
    """

    def __init__(self, window_size=200):
        self._window_size = window_size
        super(RmiNet, self).__init__()

        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(6, 32, 5, padding=4),
            torch.nn.GELU(),
            torch.nn.Conv1d(32, 32, 5, padding=4, dilation=3),
            torch.nn.GELU(),
            torch.nn.Conv1d(32, 32, 5, padding=4, dilation=3),
            torch.nn.GELU(),
            torch.nn.Conv1d(32, 1, 5, padding=4, dilation=3),
            torch.nn.GELU(),
            torch.nn.Flatten(),
        )
        self.linear_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(50),
            torch.nn.GELU(),
            torch.nn.Linear(50, 50),
            torch.nn.GELU(),
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


class DeltaTransRmiNet(BaseNet):
    """
    Convolutional neural network based RMI corrector. The network outputs a
    "delta" which is then "added" to the analytical RMIs to produce the final
    estimate.
    """

    def __init__(self, window_size=200):
        self._window_size = window_size
        super(DeltaTransRmiNet, self).__init__()

        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                6,
                16,
                7,
                padding="same",
                padding_mode="replicate",
                dilation=1,
                bias=False,
            ),
            torch.nn.BatchNorm1d(16),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(
                16,
                32,
                7,
                padding="same",
                padding_mode="replicate",
                dilation=4,
                bias=False,
            ),
            torch.nn.BatchNorm1d(32),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(
                32,
                64,
                7,
                padding="same",
                padding_mode="replicate",
                dilation=16,
                bias=False,
            ),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(
                64,
                128,
                7,
                padding="same",
                padding_mode="replicate",
                dilation=64,
                bias=False,
            ),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(
                128, 6, 7, padding="same", padding_mode="replicate", dilation=1
            ),
            torch.nn.GELU(),
        )
        self.linear_layer = torch.nn.Sequential(
            # torch.nn.LazyLinear(30),
            # torch.nn.GELU(),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(6, 6),
        )
        self.calib_mat = torch.nn.Parameter(0.05*torch.randn((6,6)))

    def forward(self, x: torch.Tensor):
        x = x[:, 1:, :]
        #x = self.norm(x)
        M = (self.I + self.calib_mat).expand((x.shape[0],6,6))
        x = torch.matmul(M, x)
        x = self.conv_layer(x)
        x = torch.mean(x, dim=2)
        x = self.linear_layer(x)
        return x


class DeltaRotRmiNet(torch.nn.Module):
    """
    Convolutional neural network based RMI corrector. The network outputs a
    "delta" which is then "added" to the analytical RMIs to produce the final
    estimate.
    """

    def __init__(self, window_size=200):
        self._window_size = window_size
        super(DeltaRotRmiNet, self).__init__()
        in_dim = 6
        out_dim = 3
        dropout = 0
        momentum = 0.1
        # channel dimension
        c0 = 16
        c1 = 2 * c0
        c2 = 2 * c1
        c3 = 2 * c2
        # kernel dimension (odd number)
        k0 = 7
        k1 = 7
        k2 = 7
        k3 = 7
        # dilation dimension
        d0 = 4
        d1 = 4
        d2 = 4
        # padding
        p0 = (k0 - 1) + d0 * (k1 - 1) + d0 * d1 * (k2 - 1) + d0 * d1 * d2 * (k3 - 1)
        # nets
        self.conv_layer = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((p0, 0)),  # padding at start
            torch.nn.Conv1d(in_dim, c0, k0, dilation=1),
            torch.nn.BatchNorm1d(c0, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c0, c1, k1, dilation=d0),
            torch.nn.BatchNorm1d(c1, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c1, c2, k2, dilation=d0 * d1),
            torch.nn.BatchNorm1d(c2, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c2, c3, k3, dilation=d0 * d1 * d2),
            torch.nn.BatchNorm1d(c3, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c3, out_dim, 1, dilation=1),
            torch.nn.Flatten(),  # no padding at end
        )
        self.linear_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(3),
            # torch.nn.GELU(),
            # torch.nn.Linear(50, 3),
        )

    def forward(self, x: torch.Tensor):
        x = x.detach().clone()
        x = x[:, 1:, :]
        x[:, 3:6, :] /= 9.80665
        x = self.conv_layer(x)
        x = self.linear_layer(x)
        return x


class RmiModel(torch.nn.Module):
    """
    Classic IMU dead-reckoning RMIs from Forster et. al. (2017) wrapped in a
    pytorch "neural network" just for easy hot-swapping and comparison.
    """

    def __init__(self, window_size=200):
        super(RmiModel, self).__init__()

    def forward(self, x):
        # shape[0] is the batch size
        y = torch.zeros((x.shape[0], 15))
        for idx in range(x.shape[0]):
            y[idx, :] = get_rmis(x[idx, :, :])

        return y


def get_rmis(x):
    """
    Computes RMIs from accel and gyro data supplied as a big torch Tensor of
    dimension [7 x N], where N is the number of measurements.

    Zeroth row of tensor is timestamps, rows 1,2,3 are gyro, rows 4,5,6 are accel.

    Unfortunately the iterative/recursive nature of this model makes it difficult
    to implement for a batch.
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
    """
    Get RMIs from the ground truth absolute poses.
    """
    g_a = torch.Tensor([0, 0, -9.80665]).view((-1, 1))
    DC = C_i.T @ C_j
    DV = C_i.T @ (v_j - v_i - DT * g_a)
    DR = C_i.T @ (r_j - r_i - v_i * DT - 0.5 * g_a * (DT ** 2))

    return DR, DV, DC
