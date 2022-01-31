from re import X
import sys
import os.path
from time import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from model import get_gt_rmis, get_rmis, get_rmi_batch
import torch
from utils import unflatten_pose
from evaluation import imu_dead_reckoning
from dataset import RmiDataset
from torch.utils.data import DataLoader


def test_rmis():
    torch.set_default_dtype(torch.float64)
    g = torch.Tensor([0, 0, -9.80665]).view((-1, 1))
    filename = "./data/processed/v1_01_easy.csv"
    data = RmiDataset(filename, window_size="full", use_cache=False)
    x, rmis, poses = data.get_item_with_poses(0)

    DT = x[0, -1] - x[0, 0]
    r_i, v_i, C_i = unflatten_pose(poses[0])
    r_j, v_j, C_j = unflatten_pose(poses[1])

    # Get RMIs from ground truth, use to get final pose.
    DR_gt, DV_gt, DC_gt = get_gt_rmis(r_i, v_i, C_i, r_j, v_j, C_j, DT)
    DR_gt2, DV_gt2, DC_gt2 = unflatten_pose(rmis)
    assert torch.allclose(DR_gt, DR_gt2)
    assert torch.allclose(DV_gt, DV_gt2)
    assert torch.allclose(DC_gt, DC_gt2)

    C_j_rmi_gt = C_i @ DC_gt
    v_j_rmi_gt = v_i + g * DT + C_i @ DV_gt
    r_j_rmi_gt = r_i + v_i * DT + 0.5 * g * (DT ** 2) + C_i @ DR_gt

    # Get RMIs from measurements, use to get final pose.
    y = get_rmis(x)
    DR, DV, DC = unflatten_pose(y)
    C_j_rmi = C_i @ DC
    v_j_rmi = v_i + g * DT + C_i @ DV
    r_j_rmi = r_i + v_i * DT + 0.5 * g * DT ** 2 + C_i @ DR

    # Get final pose by classic dead reckoning
    t = x[0, :]
    gyro = x[1:4, :]
    accel = x[4:7, :]
    traj = imu_dead_reckoning("./data/processed/v1_01_easy.csv")
    r_j_dr = traj["r"][:, -1].reshape((-1, 1))
    v_j_dr = traj["v"][:, -1].reshape((-1, 1))
    C_j_dr = traj["C"][-1]

    assert torch.allclose(C_j_rmi, C_j_dr, atol=1e-6)
    assert torch.allclose(r_j_rmi, r_j_dr, atol=1e-6)
    assert torch.allclose(v_j_rmi, v_j_dr, atol=1e-5)
    assert torch.allclose(C_j, C_j_rmi_gt, atol=1e-5)
    assert torch.allclose(r_j, r_j_rmi_gt, atol=1e-1)
    assert torch.allclose(v_j, v_j_rmi_gt, atol=1e-2)


def test_rmi_batch():

    filename = "./data/processed/v1_01_easy.csv"
    dataset = RmiDataset(filename, window_size=200, use_cache=False)
    loader = DataLoader(dataset, batch_size=10)
    x, y_gt = next(iter(loader))
    start1 = time()
    y_test = get_rmi_batch(x)
    end1 = time()

    # Alternative approach
    start2 = time()
    y = torch.zeros((x.shape[0], 15))
    for idx in range(x.shape[0]):
        y[idx, :] = get_rmis(x[idx, :, :])
    end2 = time()
    print(end1 - start1)
    print(end2 - start2)
    assert torch.allclose(y, y_test)


if __name__ == "__main__":
    test_rmis()
