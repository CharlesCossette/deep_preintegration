# %%
from math import pi
from model import *
import matplotlib.pyplot as plt
from utils import imu_dead_reckoning
from losses import pose_loss, delta_rmi_loss
import seaborn as sns

sns.set_theme(context="paper")


def test_rminet(net, filename):
    data = load_processed_data(test_file)
    t = data["timestamp"]
    r_zw_a_gt = data["r_zw_a"]
    v_zwa_a_gt = data["v_zwa_a"]
    C_ab_gt = data["C_ab"]
    g_a = torch.Tensor([0, 0, -9.80665]).reshape((-1, 1))
    N = net._window_size
    dataset = RmiDataset(filename, window_size=N, stride=N - 1)
    with torch.no_grad():
        net.eval()
        r_i = r_zw_a_gt[:, 0].reshape((-1, 1))
        v_i = v_zwa_a_gt[:, 0].reshape((-1, 1))
        C_i = C_ab_gt[:, 0].reshape((3, 3))

        t_data = [torch.Tensor([0.0])]
        r_data = [r_i]
        v_data = [v_i]
        C_data = [C_i]
        r_gt_data = [r_i]
        v_gt_data = [v_i]
        C_gt_data = [C_i]
        for i in range(len(dataset)):
            imu_window, rmis_gt, poses_gt = dataset.get_item_with_poses(i)

            # Get RMIs from neural network
            imu_window = imu_window.unsqueeze(0)
            rmis = net(imu_window[:, 1:, :])
            DR, DV, DC = unflatten_pose(rmis[0, :])
            t_i = imu_window[0, 0, 0]
            t_j = imu_window[0, 0, -1]
            DT = t_j - t_i

            # ground truth poses at endpoints
            r_gt_i, v_gt_i, C_gt_i = unflatten_pose(poses_gt[0])
            r_gt_j, v_gt_j, C_gt_j = unflatten_pose(poses_gt[1])
            r_gt_data.append(r_gt_j)
            v_gt_data.append(v_gt_j)
            C_gt_data.append(C_gt_j)

            # Get RMIs from ground truth poses
            DR_gt, DV_gt, DC_gt = get_gt_rmis(
                r_gt_i, v_gt_i, C_gt_i, r_gt_j, v_gt_j, C_gt_j, DT
            )

            # Get RMIs from measurements
            DR_dr, DV_dr, DC_dr = unflatten_pose(get_rmis(imu_window[0, :, :]))

            # Use RMIs to predict motion forward
            C_j = C_i @ DC
            v_j = v_i + g_a * DT + C_i @ DV
            r_j = r_i + v_i * DT + 0.5 * g_a * (DT ** 2) + C_i @ DR
            r_i = r_j
            v_i = v_j
            C_i = C_j

            t_data.append(t_j)
            r_data.append(r_j)
            v_data.append(v_j)
            C_data.append(C_j)

        t_data = torch.hstack(t_data)
        r_data = torch.hstack(r_data)
        v_data = torch.hstack(v_data)
        r_gt_data = torch.hstack(r_gt_data)
        v_gt_data = torch.hstack(v_gt_data)
        return {
            "t": t_data,
            "r": r_data,
            "v": v_data,
            "C": C_data,
            "r_gt": r_gt_data,
            "v_gt": v_gt_data,
            "C_gt": C_gt_data,
        }


def test_drminet(net, filename):
    g_a = torch.Tensor([0, 0, -9.80665]).reshape((-1, 1))
    N = net._window_size
    dataset = RmiDataset(filename, window_size=N, stride=N - 1, with_model=True)
    with torch.no_grad():
        net.eval()
        _, _, poses_gt = dataset.get_item_with_poses(0)
        r_i, v_i, C_i = unflatten_pose(poses_gt[0])

        t_data = [torch.Tensor([0.0])]
        r_data = [r_i]
        v_data = [v_i]
        C_data = [C_i]
        r_gt_data = [r_i]
        v_gt_data = [v_i]
        C_gt_data = [C_i]
        for i in range(len(dataset)):
            imu_window, rmis, poses_gt = dataset.get_item_with_poses(i)

            # Get RMIs from neural network
            imu_window = imu_window.unsqueeze(0)
            delta = net(imu_window)
            # print(delta_rmi_loss(delta, rmis.unsqueeze(0)))
            t_i = imu_window[0, 0, 0]
            t_j = imu_window[0, 0, -1]
            DT = t_j - t_i
            DR_meas, DV_meas, DC_meas = unflatten_pose(rmis[15:])
            dphi = delta[0, 0:3].reshape((-1, 1))
            dv = delta[0, 3:6].reshape((-1, 1))
            dr = delta[0, 6:9].reshape((-1, 1))

            DC = torch.matmul(DC_meas, SO3.Exp(dphi).squeeze())
            DV = DV_meas + dv
            DR = DR_meas + dr

            # ground truth poses at endpoints
            r_gt_i, v_gt_i, C_gt_i = unflatten_pose(poses_gt[0])
            r_gt_j, v_gt_j, C_gt_j = unflatten_pose(poses_gt[1])
            r_gt_data.append(r_gt_j)
            v_gt_data.append(v_gt_j)
            C_gt_data.append(C_gt_j)

            # Get RMIs from ground truth poses
            DR_gt, DV_gt, DC_gt = get_gt_rmis(
                r_gt_i, v_gt_i, C_gt_i, r_gt_j, v_gt_j, C_gt_j, DT
            )

            # Use RMIs to predict motion forward
            C_j = C_i @ DC
            v_j = v_i + g_a * DT + C_i @ DV
            r_j = r_i + v_i * DT + 0.5 * g_a * (DT ** 2) + C_i @ DR
            r_i = r_j
            v_i = v_j
            C_i = C_j

            t_data.append(t_j)
            r_data.append(r_j)
            v_data.append(v_j)
            C_data.append(C_j)

            net_loss = pose_loss(
                flatten_pose(DR, DV, DC).unsqueeze(0),
                flatten_pose(DR_gt, DV_gt, DC_gt).unsqueeze(0),
            )
            model_loss = pose_loss(
                flatten_pose(DR_meas, DV_meas, DC_meas).unsqueeze(0),
                flatten_pose(DR_gt, DV_gt, DC_gt).unsqueeze(0),
            )
            print("Net Loss: %.6f, Model Loss: %.6f" % (net_loss, model_loss))

        t_data = torch.hstack(t_data)
        r_data = torch.hstack(r_data)
        v_data = torch.hstack(v_data)
        r_gt_data = torch.hstack(r_gt_data)
        v_gt_data = torch.hstack(v_gt_data)
        return {
            "t": t_data,
            "r": r_data,
            "v": v_data,
            "C": C_data,
            "r_gt": r_gt_data,
            "v_gt": v_gt_data,
            "C_gt": C_gt_data,
        }


def test_seperated_nets(trans_net, rot_net, filename):
    g_a = torch.Tensor([0, 0, -9.80665]).reshape((-1, 1))
    N = trans_net._window_size

    dataset = RmiDataset(filename, window_size=N, stride=N - 1, with_model=True)
    with torch.no_grad():
        trans_net.eval()
        rot_net.eval()

        _, _, poses_gt = dataset.get_item_with_poses(0)
        r_i, v_i, C_i = unflatten_pose(poses_gt[0])

        t_data = [torch.Tensor([0.0])]
        r_data = [r_i]
        v_data = [v_i]
        C_data = [C_i]
        r_gt_data = [r_i]
        v_gt_data = [v_i]
        C_gt_data = [C_i]
        for i in range(len(dataset)):
            imu_window, rmis, poses_gt = dataset.get_item_with_poses(i)

            # Get RMIs from neural network
            imu_window = imu_window.unsqueeze(0)
            out_rot = rot_net(imu_window)
            out_trans = trans_net(imu_window)
            t_i = imu_window[0, 0, 0]
            t_j = imu_window[0, 0, -1]
            DT = t_j - t_i
            DR_meas, DV_meas, DC_meas = unflatten_pose(rmis[15:])
            dphi = out_rot[0, :].reshape((-1, 1))
            dv = out_trans[0, 0:3].reshape((-1, 1))
            dr = out_trans[0, 3:].reshape((-1, 1))

            DC = torch.matmul(DC_meas, SO3.Exp(dphi).squeeze())
            DV = DV_meas + dv
            DR = DR_meas + dr

            # ground truth poses at endpoints
            r_gt_i, v_gt_i, C_gt_i = unflatten_pose(poses_gt[0])
            r_gt_j, v_gt_j, C_gt_j = unflatten_pose(poses_gt[1])
            r_gt_data.append(r_gt_j)
            v_gt_data.append(v_gt_j)
            C_gt_data.append(C_gt_j)

            # Get RMIs from ground truth poses
            DR_gt, DV_gt, DC_gt = get_gt_rmis(
                r_gt_i, v_gt_i, C_gt_i, r_gt_j, v_gt_j, C_gt_j, DT
            )

            # Use RMIs to predict motion forward
            C_j = C_i @ DC
            v_j = v_i + g_a * DT + C_i @ DV
            r_j = r_i + v_i * DT + 0.5 * g_a * (DT ** 2) + C_i @ DR
            r_i = r_j
            v_i = v_j
            C_i = C_j

            t_data.append(t_j)
            r_data.append(r_j)
            v_data.append(v_j)
            C_data.append(C_j)

            net_loss = pose_loss(
                flatten_pose(DR, DV, DC).unsqueeze(0),
                flatten_pose(DR_gt, DV_gt, DC_gt).unsqueeze(0),
            )
            model_loss = pose_loss(
                flatten_pose(DR_meas, DV_meas, DC_meas).unsqueeze(0),
                flatten_pose(DR_gt, DV_gt, DC_gt).unsqueeze(0),
            )
            print("Net Loss: %.6f, Model Loss: %.6f" % (net_loss, model_loss))

        t_data = torch.hstack(t_data)
        r_data = torch.hstack(r_data)
        v_data = torch.hstack(v_data)
        C_data = torch.stack(C_data, 0)
        r_gt_data = torch.hstack(r_gt_data)
        v_gt_data = torch.hstack(v_gt_data)
        C_gt_data = torch.stack(C_gt_data, 0)
        return {
            "t": t_data,
            "r": r_data,
            "v": v_data,
            "C": C_data,
            "r_gt": r_gt_data,
            "v_gt": v_gt_data,
            "C_gt": C_gt_data,
        }


N = 1000  # window size
torch.set_default_dtype(torch.float64)
test_file = "./data/processed/v1_01_easy.csv"
data = load_processed_data(test_file)
t = data["timestamp"]
r_zw_a_gt = data["r_zw_a"]
v_zwa_a_gt = data["v_zwa_a"]
C_ab_gt = data["C_ab"]
g_a = torch.Tensor([0, 0, -9.80665]).reshape((-1, 1))

# Do classical dead reckoning
traj = imu_dead_reckoning(
    t,
    r_zw_a_gt[:, 0].reshape((-1, 1)),
    v_zwa_a_gt[:, 0].reshape((-1, 1)),
    C_ab_gt[:, 0].reshape((3, 3)),
    data["gyro"],
    data["accel"],
)

# #
# net = RmiNet(window_size=N)
# net.load_state_dict(torch.load("./results/rminet_weights.pt"))
# traj_net = test_rminet(net, test_file)

# net = DeltaRmiNet(window_size=N)
# net.load_state_dict(torch.load("./results/drminet_weights.pt"))
# traj_net = test_drminet(net, test_file)

trans_net = DeltaTransRmiNet(N)
rot_net = DeltaRotRmiNet(N)
trans_net.load_state_dict(torch.load("./results/dtransrminet_weights.pt"))
rot_net.load_state_dict(torch.load("./results/drotrminet_weights.pt"))
traj_net = test_seperated_nets(trans_net, rot_net, test_file)

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot(r_zw_a_gt[0, :], r_zw_a_gt[1, :], r_zw_a_gt[2, :], label="Ground truth")

# ax.plot(traj["r"][0, :], traj["r"][1, :], traj["r"][2, :], label="Analytical Model")

# ax.plot(
#     traj_net["r"][0, :], traj_net["r"][1, :], traj_net["r"][2, :], label="RMI-Net"
# )
# ax.legend()
# ax.set_xlim(-5,5)
# ax.set_ylim(-5,5)
# ax.set_zlim(-5,5)

# %%
fig, axs = plt.subplots(3, 1)
fig.suptitle("Position Trajectory")
label = ["x", "y", "z"]
for i in range(3):
    axs[i].plot(t, r_zw_a_gt[i, :], label="Ground Truth")
    axs[i].plot(traj["t"], traj["r"][i, :], label="Traditional")
    axs[i].plot(traj_net["t"], traj_net["r"][i, :], label="RMI-Net")
    axs[i].set_ylabel(label[i])

axs[-1].set_xlabel("Time (s)")
axs[0].legend()

fig, axs = plt.subplots(3, 1)
fig.suptitle("Velocity Trajectory")
label = ["x", "y", "z"]
for i in range(3):
    axs[i].plot(t, v_zwa_a_gt[i, :], label="Ground Truth")
    axs[i].plot(traj["t"], traj["v"][i, :], label="Traditional")
    axs[i].plot(traj_net["t"], traj_net["v"][i, :], label="RMI-Net")
    axs[i].set_ylabel(label[i])

axs[-1].set_xlabel("Time (s)")
axs[0].legend()

fig, axs = plt.subplots(3, 1)
fig.suptitle("Attitude Error (degrees)")
label = ["x", "y", "z"]
C_dr = torch.stack(traj["C"], 0)
C_dr_gt = C_ab_gt.T.reshape((-1, 3, 3))
e_att_net = SO3.Log(
    torch.matmul(torch.transpose(traj_net["C"], 1, 2), traj_net["C_gt"])
)
e_att = SO3.Log(torch.matmul(torch.transpose(C_dr, 1, 2), C_dr_gt))
for i in range(3):
    axs[i].plot(traj["t"], e_att[:, i] * (360 / (2 * pi)), label="Traditional")
    axs[i].plot(traj_net["t"], e_att_net[:, i] * (360 / (2 * pi)), label="RMI-Net")
    axs[i].set_ylabel(label[i])

axs[-1].set_xlabel("Time (s)")
axs[0].legend()


# %%
plt.show()
