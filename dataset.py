from math import floor
import os.path
from torch.utils.data import Dataset
import torch
import pandas as pd
from utils import flatten_pose
from model import get_gt_rmis, get_rmis

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
        accel_bias=[0, 0, 0],
        gyro_bias=None,
        use_cache=True,
    ):
        self._filename = filename
        self._data = torch.Tensor(pd.read_csv(filename, sep=",").values)

        if gyro_bias is None:
            gyro_bias = torch.mean(self._data[:1000, 1:4], 0, keepdim=False)
        else:
            gyro_bias = torch.Tensor(gyro_bias)

        accel_bias = torch.Tensor(accel_bias)
        self._data[:, 1:4] -= gyro_bias
        self._data[:, 4:7] -= accel_bias

        if window_size == "full":
            window_size = self._data.shape[0]

        self._window_size = window_size
        self._stride = stride
        self._with_model = with_model
        self._poses = None
        self._quickloader_valid = [False] * self.__len__()
        self._is_cached = False
        if with_model:
            self._quickloader_y = torch.zeros((self.__len__(), 30))
        else:
            self._quickloader_y = torch.zeros((self.__len__(), 15))

        if use_cache:
            self.load_cache()

    def __len__(self):
        return floor((self._data.shape[0] - self._window_size) / self._stride) + 1

    def __getitem__(self, idx):
        # Get from quickloader if available, otherwise compute from scratch.
        if self._quickloader_valid[idx]:
            range_start = idx * self._stride
            range_stop = range_start + self._window_size
            x = self._data[range_start:range_stop, 0:7].T
            y = self._quickloader_y[idx, :]
        else:
            x, y = self._compute_sample(idx)
            self._quickloader_y[idx, :] = y
            self._quickloader_valid[idx] = True

        if not self._is_cached:
            if all(self._quickloader_valid):
                self.save_cache()
                self._is_cached = True

        return x, y

    def _compute_sample(self, idx):
        range_start = idx * self._stride
        range_stop = range_start + self._window_size

        if range_stop > (self._data.shape[0] + 1):
            raise RuntimeError("programming error")

        # Convert to Torch tensor.
        sample_data = self._data[range_start:range_stop, :]

        # Get network input: gyro and accel data.
        t_data = sample_data[:, 0]
        x = self._data[range_start:range_stop, 0:7].T

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
            print(str(idx) + ": Computing RMI analytically from scratch.")
            y = torch.hstack((y, get_rmis(x)))

        # Store absolute poses for convenience
        self._poses = [
            flatten_pose(r_i, v_i, C_i),
            flatten_pose(r_j, v_j, C_j),
        ]
        return x, y

    def get_item_with_poses(self, idx):
        """
        Provides a sample with an additional third output: the absolute poses
        at the beginning and end of the window.
        """
        # TODO: Doesnt work with quickloader!!
        x, y = self.__getitem__(idx)
        return x, y, self._poses

    def get_index_of_time(self, t):
        t_data = self._data[:, 0]
        diff = torch.abs(t_data - t)
        idx_data = torch.argmin(diff)
        idx = ((idx_data / diff.shape[0]) * self.__len__()).int().item()
        return idx

    def reset(self):
        """
        Clears the internal quickloader so that all samples must be computed
        from scratch.
        """
        if self._with_model:
            self._quickloader_y = torch.zeros((self.__len__(), 30))
        else:
            self._quickloader_y = torch.zeros((self.__len__(), 15))

    def cache_filename(self):
        name = self._filename.split("/")[-1]
        name = name.split(".")[0]

        cachefile = (
            "./data/cache/"
            + str(name)
            + "_"
            + str(self._window_size)
            + "_"
            + str(self._stride)
            + "_"
            + str(self._with_model)
            + ".csv"
        )
        return cachefile

    def save_cache(self):
        cachefile = self.cache_filename()
        pd.DataFrame(self._quickloader_y.numpy()).to_csv(
            cachefile, header=False, index=False
        )

    def load_cache(self):
        cachefile = self.cache_filename()
        if os.path.isfile(cachefile):
            self._is_cached = True
            self._quickloader_valid = [True] * self.__len__()
            self._quickloader_y = torch.Tensor(
                pd.read_csv(cachefile, header=None).values
            )
            
def add_noise(x):
    """Add Gaussian noise and bias to input"""
    imu = x[:,1:,:]

    # noise density
    imu_std = torch.Tensor([8e-5, 1e-3], device=x.device)
    # bias repeatability (without in-run bias stability)
    # imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
    # uni = torch.distributions.uniform.Uniform(-torch.ones(1),
    #     torch.ones(1))

    noise = torch.randn_like(imu, device = x.device)
    noise[:, :3, :] = noise[:, :3, :] * imu_std[0]
    noise[:, 3:6, :] = noise[:, 3:6, :] * imu_std[1]

    # # bias repeatability (without in run bias stability)
    # b0 = self.uni.sample(x[:, 0].shape).to(x.device)
    # b0[:, :3, :] = b0[:,:3,:] * self.imu_b0[0]
    # b0[:, 3:6, :] =  b0[:, 3:6,:] * self.imu_b0[1]
    x[:,1:,:] = imu + noise
    return x