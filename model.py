from tkinter import Y
import torch
from pylie.torch import SO3
from utils import flatten_pose, bmv


class BaseNet(torch.nn.Module):
    """
    Base neural network which contains some general functions such as input
    normalization.
    """

    def __init__(self):
        super(BaseNet, self).__init__()

        # for normalizing inputs
        self.mean_x = torch.Tensor(
            [4.2766e-02, 8.1577e-03, -1.5818e-02, 9.2993e00, -8.7913e-02, -3.3231e00]
        ).view((-1, 1))
        self.std_x = torch.Tensor(
            [0.4142, 0.3522, 0.2881, 1.4740, 0.6553, 0.8728]
        ).view((-1, 1))

        # These are added as parameters so that they are automatically
        # moved to the GPU by net.to("cuda")
        self.mean_x = torch.nn.Parameter(self.mean_x, requires_grad=False)
        self.std_x = torch.nn.Parameter(self.std_x, requires_grad=False)
        self.I = torch.nn.Parameter(torch.eye(6), requires_grad=False)
        self.calib_mat = torch.nn.Parameter(0.05 * torch.randn((6, 6)))
        self.bias = torch.nn.Parameter(0.01 * torch.randn((6, 1)))

    def norm(self, x):
        return (x - self.mean_x) / self.std_x
    
    def calibrate(self, x):
        M = (self.I + self.calib_mat).expand((x.shape[0], 6, 6))
        return torch.matmul(M, x) + self.bias

    def set_normalized_factors(self, mean_x, std_x, requires_grad=False):
        self.mean_u = torch.nn.Parameter(mean_x, requires_grad=requires_grad)
        self.std_u = torch.nn.Parameter(std_x, requires_grad=requires_grad)

    def set_calibration(self, mat = None, bias = None, requires_grad=False):
        if mat is None:
            mat = torch.zeros((6,6))
        if bias is None:
            bias = torch.zeros((6,1))

        self.calib_mat = torch.nn.Parameter(mat, requires_grad=requires_grad)
        self.bias = torch.nn.Parameter(bias.view((6,1)), requires_grad=requires_grad)


class RmiNet(BaseNet):
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
        imu = x[:, 1:, :]
        M = (self.I + self.calib_mat).expand((x.shape[0], 6, 6))
        x = torch.matmul(M, imu) + self.bias
        x = self.conv_layer(self.norm(x))
        x = self.linear_layer(x) * 100

        # Normalize the rotation matrix to make it a valid element of SO(3)
        R = torch.reshape(x[:, 0:9], (x.shape[0], 3, 3))
        U, _, VT = torch.linalg.svd(R)
        S = torch.eye(3, device=x.device).reshape((1, 3, 3)).repeat(x.shape[0], 1, 1)
        S[:, 2, 2] = torch.det(torch.matmul(U, VT))
        R_norm = torch.matmul(U, torch.matmul(S, VT))
        R_flat = torch.reshape(R_norm, (x.shape[0], 9))

        return torch.cat((R_flat, x[:, 9:]), 1)


class RmiNet2(BaseNet):
    """
    Convolutional neural network based RMI corrector. The network outputs a
    "delta" which is then "added" to the analytical RMIs to produce the final
    estimate.
    """

    def __init__(self, window_size=200, output_std_dev =  torch.Tensor([1,1,1])):
        self._window_size = window_size
        super(RmiNet2, self).__init__()

        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(6,12,7,1, padding= "same", padding_mode="replicate"),
            torch.nn.GELU(),
            torch.nn.Dropout(0.25),
            torch.nn.Conv1d(12,24,7,1,dilation=4,padding= "same",  padding_mode="replicate"),
            torch.nn.GELU(),
            torch.nn.Dropout(0.25),
            torch.nn.Conv1d(24,3,7,1,dilation=16,padding= "same",  padding_mode="replicate"),
            torch.nn.GELU(),
        )
        # self.linear_layer = torch.nn.Sequential(
        #     torch.nn.LazyLinear(10),
        #     torch.nn.GELU(), 
        #     torch.nn.Dropout(0.1),
        #     torch.nn.Linear(10, 3),
        # )
        self._output_std_dev = torch.nn.Parameter(output_std_dev, requires_grad=False)

    def forward(self, x: torch.Tensor):
        x = x.clone()
        imu = x[:, 1:, :]
        x1 = self.calibrate(imu)
        x2 = self.conv_layer(self.norm(x1))
        x2 = torch.mean(x2, dim=2)
        return self._output_std_dev*x2


class RmiModel(BaseNet):
    """
    Classic IMU dead-reckoning RMIs from Forster et. al. (2017) wrapped in a
    pytorch "neural network" just for easy hot-swapping and comparison.
    """

    def __init__(self, output_window=200):
        super(RmiModel, self).__init__()
        self.calib_mat = torch.nn.Parameter(0.01 * torch.randn((6, 6)))
        self.bias = torch.nn.Parameter(0.01 * torch.randn((6, 1)))
        self._ow = output_window

    def forward(self, x):
        x = x[:,:,-self._ow:]
        t = x[:, 0, :].unsqueeze(1)
        imu = x[:, 1:, :]
        x1 = self.calibrate(imu)
        return get_rmi_batch(torch.concat((t, x1), dim=1))


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


def get_rmi_batch(x, flatten=True):
    """
    x is now of shape [B x 7 x N]
    """
    dim_batch = x.shape[0]
    t = x[:, 0, :]
    gyro = x[:, 1:4, :]
    accel = x[:, 4:7, :]
    DC = torch.eye(3, device=x.device).expand((dim_batch, 3, 3))
    DV = torch.zeros((dim_batch, 3), device=x.device)
    DR = torch.zeros((dim_batch, 3), device=x.device)
    for idx in range(1, x.shape[2]):
        dt = (t[:, idx] - t[:, idx - 1]).unsqueeze(1)
        w = gyro[:, :, idx - 1]
        a = accel[:, :, idx - 1]
        DR = DR + DV * dt + 0.5 * bmv(DC, a) * (dt ** 2)
        DV = DV + bmv(DC, a * dt)
        DC = torch.matmul(DC, SO3.Exp(w * dt))

    if flatten:
        return flatten_pose(DR, DV, DC)
    else:
        return DR, DV, DC


def get_gt_rmis(r_i, v_i, C_i, r_j, v_j, C_j, DT):
    """
    Get RMIs from the ground truth absolute poses.
    """
    g_a = torch.Tensor([0, 0, -9.80665]).view((-1, 1))
    DC = C_i.T @ C_j
    DV = C_i.T @ (v_j - v_i - DT * g_a)
    DR = C_i.T @ (r_j - r_i - v_i * DT - 0.5 * g_a * (DT ** 2))

    return DR, DV, DC
