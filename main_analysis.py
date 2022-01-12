from model import *
import matplotlib.pyplot as plt
from utils import imu_dead_reckoning

# import seaborn as sns
# sns.set_theme(context="paper")


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
            C_j = C_i @ DC_gt
            v_j = v_i + g_a * DT + C_i @ DV_gt
            r_j = r_i + v_i * DT + 0.5 * g_a * (DT ** 2) + C_i @ DR_gt
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


if __name__ == "__main__":
    N = 400  # window size
    test_file = "./data/processed/v1_01_easy.csv"
    data = load_processed_data(test_file)
    t = data["timestamp"]
    r_zw_a_gt = data["r_zw_a"]
    v_zwa_a_gt = data["v_zwa_a"]
    C_ab_gt = data["C_ab"]
    g_a = torch.Tensor([0, 0, -9.80665]).reshape((-1, 1))

    traj = imu_dead_reckoning(
        t,
        r_zw_a_gt[:, 0].reshape((-1, 1)),
        v_zwa_a_gt[:, 0].reshape((-1, 1)),
        C_ab_gt[:, 0].reshape((3, 3)),
        data["gyro"],
        data["accel"],
    )

    net = RmiNet(window_size=N)
    net.load_state_dict(torch.load("./results/rminet_weights.pt"))
    traj_net = test_rminet(net, test_file)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot(r_zw_a_gt[0, :], r_zw_a_gt[1, :], r_zw_a_gt[2, :], label="Ground truth")

    ax.plot(traj["r"][0, :], traj["r"][1, :], traj["r"][2, :], label="Analytical Model")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    ax.plot(
        traj_net["r"][0, :], traj_net["r"][1, :], traj_net["r"][2, :], label="RMI-Net"
    )
    ax.legend()

    e_r_model = r_zw_a_gt - traj["r"]
    e_r_net = traj_net["r_gt"] - traj_net["r"]
    rmse_r_model = torch.sqrt(torch.sum(torch.pow(e_r_model, 2), 0) / 3)
    rmse_r_net = torch.sqrt(torch.sum(torch.pow(e_r_net, 2), 0) / 3)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(traj["t"], rmse_r_model, label="Analytical Model")
    ax.plot(traj_net["t"], rmse_r_net, label="RMI-Net")
    ax.legend()
    plt.show()
