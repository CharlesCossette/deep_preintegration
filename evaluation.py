from math import sqrt
from model import (
    get_gt_rmis,
    get_rmi_batch,
    get_rmis,
)
from training import valid_loop
from utils import unflatten_pose, flatten_pose
from losses import  pose_loss
import torch
from torch.utils.data import DataLoader
from pylie.torch import SO3
from dataset import RmiDataset


def imu_dead_reckoning(filename):
    g_a = torch.Tensor([0, 0, -9.80665]).reshape((-1, 1))
    dataset = RmiDataset(filename, window_size=2, stride=1, use_cache=False)
    _, _, poses = dataset.get_item_with_poses(0)
    r, v, C = unflatten_pose(poses[0])
    g = torch.Tensor([0, 0, -9.80665]).view((-1, 1))
    t_data = [torch.Tensor([0.0])]
    r_data = [r]
    v_data = [v]
    C_data = [C]
    for x, _ in dataset:
        t = x[0, :]
        w = x[1:4, 0].reshape((-1, 1))
        a = x[4:, 0].reshape((-1, 1))

        dt = t[1] - t[0]
        r = r + v * dt + 0.5 * g * (dt ** 2) + 0.5 * C @ a * (dt ** 2)
        v = v + g * dt + C @ a * dt
        C = C @ SO3.Exp(dt * w).squeeze()

        t_data.append(t[1])
        r_data.append(r)
        v_data.append(v)
        C_data.append(C)

    t_data = torch.hstack(t_data)
    r_data = torch.hstack(r_data)
    v_data = torch.hstack(v_data)
    C_data = torch.stack(C_data, 0)
    return {"t": t_data, "r": r_data, "v": v_data, "C": C_data}


def rminet_test(net, filename):
    g_a = torch.Tensor([0, 0, -9.80665]).reshape((-1, 1))
    N = net._window_size
    dataset = RmiDataset(filename, window_size=N, stride=N - 1)
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


def drminet_test(net, filename):
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


def seperated_nets_test(trans_net, rot_net, filename):
    g_a = torch.Tensor([0, 0, -9.80665]).reshape((-1, 1))
    N = trans_net._window_size

    dataset = RmiDataset(
        filename, window_size=N, stride=N - 1, with_model=True, use_cache=False
    )
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
            C_j = C_i @ DC_meas
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
                flatten_pose(DR, DV, DC_gt).unsqueeze(0),
                flatten_pose(DR_gt, DV_gt, DC_gt).unsqueeze(0),
            )
            model_loss = pose_loss(
                flatten_pose(DR_meas, DV_meas, DC_gt).unsqueeze(0),
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


def trans_net_violin(net, filename):
    N = net._window_size
    dataset = RmiDataset(
        filename, window_size=N, stride=20, with_model=True, use_cache=True
    )
    loader = DataLoader(dataset, batch_size=1)
    net.to("cpu")
    net.eval()
    r_rmse = []
    v_rmse = []
    r_meas_rmse = []
    v_meas_rmse = []
    with torch.no_grad():
        for i, validation_sample in enumerate(loader, 0):
            x, y = validation_sample

            y_hat = net(x)
            loss, info = delta_trans_rmi_loss(y_hat, y, with_info=True)
            r_rmse.append(sqrt(info["r_loss"] / 3))
            r_meas_rmse.append(sqrt(info["r_loss_meas"] / 3))
            v_rmse.append(sqrt(info["v_loss"] / 3))
            v_meas_rmse.append(sqrt(info["v_loss_meas"] / 3))

    return torch.vstack(
        [
            torch.Tensor(r_rmse),
            torch.Tensor(r_meas_rmse),
            torch.Tensor(v_rmse),
            torch.Tensor(v_meas_rmse),
        ]
    )


def rmi_estimator_test(net, filename, window_size):
    dataset = RmiDataset(
        filename, window_size=window_size, stride=100, with_model=False, use_cache=False
    )
    loader = DataLoader(dataset, batch_size=len(dataset))
    net.to("cpu")
    net.eval()
    r_rmse = []
    v_rmse = []
    C_rmse = []
    r_meas_rmse = []
    v_meas_rmse = []
    C_meas_rmse = []
    with torch.no_grad():
    
        x, y = next(iter(loader))

        y_hat = net(x)
        y_test = get_rmi_batch(x)

        for i in range(y_hat.shape[0]):
            y_hat_sample = y_hat[i,:].unsqueeze(0)
            y_test_sample = y_test[i,:].unsqueeze(0)
            y_sample = y[i,:].unsqueeze(0)
            loss, info = pose_loss(y_hat_sample, y_sample, with_info=True)
            r_rmse.append(sqrt(info["r_se"] / 3))
            v_rmse.append(sqrt(info["v_se"] / 3))
            C_rmse.append(sqrt(info["C_se"] / 3))

            loss, info = pose_loss(y_test_sample, y_sample, with_info=True)
            r_meas_rmse.append(sqrt(info["r_se"] / 3))
            v_meas_rmse.append(sqrt(info["v_se"] / 3))
            C_meas_rmse.append(sqrt(info["C_se"] / 3))

    return torch.vstack(
        [
            torch.Tensor(r_rmse),
            torch.Tensor(r_meas_rmse),
            torch.Tensor(v_rmse),
            torch.Tensor(v_meas_rmse),
            torch.Tensor(C_rmse),
            torch.Tensor(C_meas_rmse),
        ]
    )
