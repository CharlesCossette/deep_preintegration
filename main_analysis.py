from model import *
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from pylie import SO3


def imu_dead_reckoning(t, r0, v0, C0, gyro, accel):
    r = r0
    v = v0
    C = C0
    g = np.array([0, 0, -9.80665]).reshape((-1, 1))
    t_data = [0]
    r_data = [r]
    v_data = [v]
    C_data = [C]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        w = gyro[:, i - 1].reshape((-1, 1))
        a = accel[:, i - 1].reshape((-1, 1))
        r = r + v * dt + 0.5 * g * (dt ** 2) + 0.5 * C @ a * (dt ** 2)
        v = v + g * dt + C @ a * dt
        C = C @ SO3.Exp(dt * w)
        t_data.append(t[i])
        r_data.append(r)
        v_data.append(v)
        C_data.append(C)

    t_data = np.hstack(t_data)
    r_data = np.hstack(r_data)
    v_data = np.hstack(v_data)
    return {"t": t_data, "r": r_data, "v": v_data, "C": C_data}


def test_rminet(net, filename):
    data = load_processed_data(test_file)
    t = data["timestamp"]
    r_zw_a_gt = data["r_zw_a"]
    v_zwa_a_gt = data["v_zwa_a"]
    C_ab_gt = data["C_ab"]
    g_a = np.array([0, 0, -9.80665]).reshape((-1, 1))
    gyro = data["gyro"]
    accel = data["accel"]
    N = net._window_size

    with torch.no_grad():
        net.eval()
        r_j = r_zw_a_gt[:, 0].reshape((-1, 1))
        v_j = v_zwa_a_gt[:, 0].reshape((-1, 1))
        C_j = C_ab_gt[:, 0].reshape((3, 3))

        t_data = [0]
        r_data = [r_j]
        v_data = [v_j]
        C_data = [C_j]
        for i in range(N, r_zw_a_gt.shape[1], N):
            gyro_window = gyro[:, i - N : i]
            accel_window = accel[:, i - N : i]
            imu_window = torch.from_numpy(np.vstack((gyro_window, accel_window)))
            imu_window = torch.unsqueeze(imu_window, 0)
            imu_window = imu_window.to(torch.float32)
            rmis = net(imu_window)
            DC = torch.reshape(rmis[0, 0:9], (3, 3)).numpy()
            DV = torch.reshape(rmis[0, 9:12], (-1, 1)).numpy()
            DR = torch.reshape(rmis[0, 12:15], (-1, 1)).numpy()
            DT = t[i] - t[i - 200]
            r_i = r_j
            v_i = v_j
            C_i = C_j
            C_j = C_i @ DC
            v_j = v_i + g_a * DT + C_i @ DV
            r_j = r_i + v_i * DT + 0.5 * g_a * DT ** 2 + C_i @ DR
            t_data.append(t[i])
            r_data.append(r_j)
            v_data.append(v_j)
            C_data.append(C_j)

            r_i = r_zw_a_gt[:, i - N].reshape((-1, 1))
            v_i = v_zwa_a_gt[:, i - N].reshape((-1, 1))
            C_i = C_ab_gt[:, i - N].reshape((3, 3))
            r_j = r_zw_a_gt[:, i].reshape((-1, 1))
            v_j = v_zwa_a_gt[:, i].reshape((-1, 1))
            C_j = C_ab_gt[:, i].reshape((3, 3))
            DR_gt, DV_gt, DC_gt = get_gt_rmis(r_i, v_i, C_i, r_j, v_j, C_j, DT)
        t_data = np.hstack(t_data)
        r_data = np.hstack(r_data)
        v_data = np.hstack(v_data)
        return {"t": t_data, "r": r_data, "v": v_data, "C": C_data}


N = 200  # window size
test_file = "./data/processed/v1_02_medium.csv"
data = load_processed_data(test_file)
t = data["timestamp"]
r_zw_a_gt = data["r_zw_a"]
v_zwa_a_gt = data["v_zwa_a"]
C_ab_gt = data["C_ab"]
g_a = np.array([0, 0, -9.80665]).reshape((-1, 1))

traj = imu_dead_reckoning(
    t,
    r_zw_a_gt[:, 0].reshape((-1, 1)),
    v_zwa_a_gt[:, 0].reshape((-1, 1)),
    C_ab_gt[:, 0].reshape((3, 3)),
    data["gyro"],
    data["accel"],
)

net = RmiNet(window_size=N)
net.load_state_dict(torch.load("./results/temp.pt"))
traj_net = test_rminet(net, test_file)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot(r_zw_a_gt[0, :], r_zw_a_gt[1, :], r_zw_a_gt[2, :], label="Ground truth")

ax.plot(traj["r"][0, :], traj["r"][1, :], traj["r"][2, :], label="Analytical Model")
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

ax.plot(traj_net["r"][0, :], traj_net["r"][1, :], traj_net["r"][2, :], label="RMI-Net")
ax.legend()

e_r_model = r_zw_a_gt - traj["r"]
e_r_net = r_zw_a_gt - traj_net["r"]
plt.show()
