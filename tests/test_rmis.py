import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from model import RmiDataset, get_gt_rmis, get_rmis
import torch
from utils import imu_dead_reckoning, unflatten_pose


def test_rmis():
    g = torch.Tensor([0, 0, -9.80665]).view((-1, 1))
    data = RmiDataset("./data/processed/v1_01_easy.csv", window_size=200)
    x, rmis, poses = data.get_item_with_poses(15)

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
    traj = imu_dead_reckoning(t, r_i, v_i, C_i, gyro, accel)
    r_j_dr = traj["r"][:, -1].reshape((-1, 1))
    v_j_dr = traj["v"][:, -1].reshape((-1, 1))
    C_j_dr = traj["C"][-1]

    assert torch.allclose(C_j_rmi, C_j_dr, atol=1e-6)
    assert torch.allclose(r_j_rmi, r_j_dr, atol=1e-6)
    assert torch.allclose(v_j_rmi, v_j_dr, atol=1e-5)
    assert torch.allclose(C_j, C_j_rmi_gt, atol=1e-5)
    assert torch.allclose(r_j, r_j_rmi_gt, atol=1e-4)
    assert torch.allclose(v_j, v_j_rmi_gt, atol=1e-4)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    test_rmis()
