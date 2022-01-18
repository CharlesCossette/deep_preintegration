from math import sqrt
from pylie.torch import SO3
from utils import unflatten_pose, flatten_pose
import torch
from torch.nn.functional import mse_loss, huber_loss
from torch.utils.tensorboard import SummaryWriter

# TODO: recycle code. have pose_loss call the other two functions.


class CustomLoss:
    def __call__(self, y1, y2):
        raise NotImplementedError()

    def write_info(self, writer, epoch, tag):
        pass


def pose_loss(y, y_gt, with_info=False):
    """
    Computes the MSE between two flattened poses. For the attitude component,
    the error is computed as

    e = Log(C.T @ C_gt)

    where C_gt = ground truth attitude.
    """
    DR, DV, DC = unflatten_pose(y)
    DR_gt, DV_gt, DC_gt = unflatten_pose(y_gt)
    e_phi = SO3.Log(torch.matmul(torch.transpose(DC, 1, 2), DC_gt))
    e_v = DR - DR_gt
    e_r = DV - DV_gt
    e = torch.hstack((e_phi, e_v, e_r))

    # Optionally, return the component-wise loss as an info dictionary.
    if with_info:
        info = {
            "C_loss": torch.sqrt(mse_loss(e_phi, torch.zeros(e_phi.shape))),
            "v_loss": torch.sqrt(mse_loss(e_v, torch.zeros(e_v.shape))),
            "r_loss": torch.sqrt(mse_loss(e_r, torch.zeros(e_r.shape))),
        }
        return mse_loss(e, torch.zeros(e.shape)), info

    return mse_loss(e, torch.zeros(e.shape))


class PoseLoss(CustomLoss):
    def __init__(self):
        self.running_C_loss = 0.0
        self.running_v_loss = 0.0
        self.running_r_loss = 0.0

    def __call__(self, y1, y2):
        loss, info = pose_loss(y1, y2, with_info=True)
        self.running_C_loss += info["C_loss"]
        self.running_v_loss += info["v_loss"]
        self.running_r_loss += info["r_loss"]
        return loss

    def write_info(self, writer: SummaryWriter, epoch, tag=""):
        writer.add_scalar("Loss/Position" + tag, self.running_r_loss, epoch)
        writer.add_scalar("Loss/Velocity" + tag, self.running_v_loss, epoch)
        writer.add_scalar("Loss/Rotation" + tag, self.running_C_loss, epoch)
        self.running_C_loss = 0.0
        self.running_v_loss = 0.0
        self.running_r_loss = 0.0


def delta_rmi_loss(y, y_train, with_info=False):
    """
    PARAMETERS:
    -----------
    y: [N x 9]
        predicted correction to RMI from neural network
    y_train [N x 15*2]
        ground truth RMIs hstacked with analytical RMIs

    """
    dim_batch = y.shape[0]
    dphi = y[:, 0:3]
    dv = y[:, 3:6]
    dr = y[:, 6:9]

    y_gt = y_train[:, 0:15]
    DR_meas, DV_meas, DC_meas = unflatten_pose(y_train[:, 15:])

    DC_final = torch.matmul(DC_meas, SO3.Exp(dphi))
    DV_final = DV_meas + dv
    DR_final = DR_meas + dr
    y_final = torch.hstack(
        (
            DC_final.view((dim_batch, -1)),
            DV_final.view((dim_batch, -1)),
            DR_final.view((dim_batch, -1)),
        )
    )

    return pose_loss(y_final, y_gt, with_info=with_info)


class DeltaRmiLoss(CustomLoss):
    def __init__(self):
        self.running_C_loss = 0.0
        self.running_v_loss = 0.0
        self.running_r_loss = 0.0

    def __call__(self, y1, y2):
        loss, info = delta_rmi_loss(y1, y2, with_info=True)
        self.running_C_loss += info["C_loss"]
        self.running_v_loss += info["v_loss"]
        self.running_r_loss += info["r_loss"]
        return loss

    def write_info(self, writer: SummaryWriter, epoch, tag=""):
        writer.add_scalar("Loss/Position" + tag, self.running_r_loss, epoch)
        writer.add_scalar("Loss/Velocity" + tag, self.running_v_loss, epoch)
        writer.add_scalar("Loss/Rotation" + tag, self.running_C_loss, epoch)
        self.running_C_loss = 0.0
        self.running_v_loss = 0.0
        self.running_r_loss = 0.0


def delta_trans_rmi_loss(y, y_train, with_info=False):
    """
    PARAMETERS:
    -----------
    y: [N x 6]
        predicted correction to RMI from neural network
    y_train [N x 15*2]
        ground truth RMIs hstacked with analytical RMIs

    """

    y_gt = y_train[:, 0:15]
    y_meas = y_train[:, 15:]
    y_final = y + y_meas[:, 9:]
    loss = huber_loss(y_final, y_gt[:, 9:], delta=0.1)

    if with_info:
        vel_rmse = mse_loss(y_final[:, 0:3], y_gt[:, 9:12], reduction="sum")
        pos_rmse = mse_loss(y_final[:, 3:], y_gt[:, 12:], reduction="sum")
        vel_meas_rmse = mse_loss(y_meas[:, 9:12], y_gt[:, 9:12], reduction="sum")
        pos_meas_rmse = mse_loss(y_meas[:, 12:], y_gt[:, 12:], reduction="sum")
        return loss, {
            "v_loss": vel_rmse.item(),
            "r_loss": pos_rmse.item(),
            "v_loss_meas": vel_meas_rmse.item(),
            "r_loss_meas": pos_meas_rmse.item(),
        }

    return loss


class DeltaTransRmiLoss(CustomLoss):
    def __init__(self):
        self.running_v_loss = 0.0
        self.running_r_loss = 0.0
        self.running_v_loss_meas = 0.0
        self.running_r_loss_meas = 0.0
        self.total_samples = 0

    def __call__(self, y1, y2):
        loss, info = delta_trans_rmi_loss(y1, y2, with_info=True)
        self.running_v_loss += info["v_loss"]
        self.running_r_loss += info["r_loss"]
        self.running_v_loss_meas += info["v_loss_meas"]
        self.running_r_loss_meas += info["r_loss_meas"]
        self.total_samples += y1.shape[0]
        return loss

    def write_info(self, writer: SummaryWriter, epoch, tag=""):
        N = self.total_samples
        writer.add_scalars(
            "RMSE/Velocity/" + tag,
            {
                tag: sqrt(self.running_v_loss / N),
                "Model": sqrt(self.running_v_loss_meas / N),
            },
            epoch,
        )
        writer.add_scalars(
            "RMSE/Position/" + tag,
            {
                tag: sqrt(self.running_r_loss / N),
                "Model": sqrt(self.running_r_loss_meas / N),
            },
            epoch,
        )
        self.running_v_loss = 0.0
        self.running_r_loss = 0.0
        self.running_v_loss_meas = 0.0
        self.running_r_loss_meas = 0.0
        self.total_samples = 0


def delta_rot_rmi_loss(y, y_train, with_info=False):
    """
    PARAMETERS:
    -----------
    y: [N x 3]
        predicted correction to RMI from neural network
    y_train [N x 15*2]
        ground truth RMIs hstacked with analytical RMIs

    """
    # TODO: could get cleaned up a little bit.
    dim_batch = y.shape[0]
    dphi = y

    y_gt = y_train[:, 0:15]
    y_meas = y_train[:, 15:]
    _, _, DC_meas = unflatten_pose(y_meas)

    DC_final = torch.matmul(DC_meas, SO3.Exp(dphi))
    _, _, DC_gt = unflatten_pose(y_gt)

    e_phi = SO3.Log(torch.matmul(torch.transpose(DC_final, 1, 2), DC_gt))

    mse = mse_loss(e_phi, torch.zeros(e_phi.shape, device=y.device))
    scaled_loss = mse_loss(e_phi, torch.zeros(e_phi.shape, device=y.device))
    if with_info:
        e_phi_meas = SO3.Log(torch.matmul(torch.transpose(DC_meas, 1, 2), DC_gt))
        mse_meas = mse_loss(
            e_phi_meas, torch.zeros(e_phi_meas.shape, device=e_phi_meas.device)
        )
        return scaled_loss, {
            "C_loss": torch.sqrt(mse).item(),
            "C_loss_meas": torch.sqrt(mse_meas).item(),
        }

    return scaled_loss


class DeltaRotRmiLoss(CustomLoss):
    def __init__(self):
        self.running_C_loss = 0.0
        self.running_C_loss_meas = 0.0

    def __call__(self, y1, y2):
        loss, info = delta_rot_rmi_loss(y1, y2, with_info=True)
        self.running_C_loss += info["C_loss"]
        self.running_C_loss_meas += info["C_loss_meas"]
        return loss

    def write_info(self, writer: SummaryWriter, epoch, tag=""):
        writer.add_scalars(
            "RMSE/Rotation/" + tag,
            {tag: self.running_C_loss, "Model": self.running_C_loss_meas},
            epoch,
        )
        self.running_C_loss = 0.0
        self.running_C_loss_meas = 0.0
