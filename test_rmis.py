from tkinter import X
from model import *
from utils import imu_dead_reckoning


def test_rmis():
    g = np.array([0, 0, -9.80665]).reshape((-1, 1))
    data = RmiDataset("./data/processed/v1_01_easy.csv")
    x, rmis, poses = data.get_item_with_poses(0)

    y = get_rmis(x)
    y = y.detach().to("cpu").numpy().reshape((1, -1))
    DR, DV, DC = unflatten_pose(y)

    DT = poses[1][0] - poses[0][0]
    r_i = poses[0][1]
    v_i = poses[0][2]
    C_i = poses[0][3]
    C_j = C_i @ DC
    v_j = v_i + g * DT + C_i @ DV
    r_j = r_i + v_i * DT + 0.5 * g * DT ** 2 + C_i @ DR

    t = x[0, :].detach().to("cpu").numpy()
    gyro = x[1:4, :].detach().to("cpu").numpy()
    accel = x[4:7, :].detach().to("cpu").numpy()
    traj = imu_dead_reckoning(t, poses[0][1], poses[0][2], poses[0][3], gyro, accel)
    r_test = traj["r"][:, -1].reshape((-1,1))
    v_test = traj["v"][:, -1].reshape((-1,1))
    C_test = traj["C"][-1]

    assert np.allclose(C_j, C_test, rtol=1e-4)
    assert np.allclose(r_j, r_test)
    assert np.allclose(v_j, v_test, rtol=1e-3)


if __name__ == "__main__":
    test_rmis()
