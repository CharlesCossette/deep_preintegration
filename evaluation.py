from model import (
    RmiDataset,
    get_gt_rmis,
    get_rmis,
)
from utils import unflatten_pose, flatten_pose
from losses import pose_loss
import torch
from pylie.torch import SO3


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
            C_j = C_i @ DC_gt
            v_j = v_i + g_a * DT + C_i @ DV_meas
            r_j = r_i + v_i * DT + 0.5 * g_a * (DT ** 2) + C_i @ DR_meas
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
