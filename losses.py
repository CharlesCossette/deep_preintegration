from math import sqrt
from pylie.torch import SO3
from utils import unflatten_pose
import torch
from torch.nn.functional import mse_loss, huber_loss
from torch.utils.tensorboard import SummaryWriter


def pose_loss(y, y_gt, with_info=False):
    """
    Computes the the loss between two flattened poses. For the attitude component,
    the error is computed as

    e = Log(C.T @ C_gt)

    where C_gt = ground truth attitude.
    """
    C_flat = y[:, 0:9]
    v = y[:,9:12]
    r = y[:,12:]
    if with_info:
        C_loss, C_info = attitude_loss(C_flat, y_gt, with_info)
        v_loss, v_info = velocity_loss(v, y_gt, with_info)
        r_loss, r_info = position_loss(r, y_gt, with_info)
        loss = C_loss + v_loss + r_loss
        info = {**C_info, **v_info, **r_info}
        return loss, info 
    else:
        C_loss = attitude_loss(C_flat, y_gt, with_info)
        v_loss = velocity_loss(v, y_gt, with_info)
        r_loss = position_loss(r, y_gt, with_info)
        loss = C_loss + v_loss + r_loss
        return loss 
    



def attitude_loss(y, y_gt, with_info=False, delta=0.005, std_dev=0.01):
    """
    Attitude loss function with error as

    e = Log(C.T @ C_gt)

    where C_gt = ground truth attitude.
    """
    dim_batch = y.shape[0]
    DC = y.view((dim_batch, 3, 3))
    _, _, DC_gt = unflatten_pose(y_gt)
    e_phi = SO3.Log(torch.matmul(torch.transpose(DC, 1, 2), DC_gt))
    loss = huber_loss(
        e_phi / std_dev, torch.zeros(e_phi.shape, device=y.device), delta=delta
    )
    if with_info:
        # Return sum of squared errors.
        info = {
            "C_se": mse_loss(
                e_phi, torch.zeros(e_phi.shape, device=y.device), reduction="sum"
            ).item(),
            "C_loss":loss.item()
        }
        return loss, info

    return loss


def velocity_loss(y, y_gt, with_info=False, delta=1, std_dev=1):
    """
    Computes the velocity loss component between two flattened poses.
    """
    if y.shape[1] == 15:
        DV = y[:,9:12]
    elif y.shape[1] == 9:
        DV = y[:,3:6]
    elif y.shape[1] == 3:
        DV = y
    else:
        raise RuntimeError("Input 1 shape 1 must either be 15, 9 or 3.")

    if y_gt.shape[1] == 15:
        _, DV_gt, _ = unflatten_pose(y_gt)
    elif y_gt.shape[1] == 9:
        DV_gt = y_gt[:,3:6]
    else:
        raise RuntimeError("Input 2 shape 1 must either be 15, 9.")

    e_v = DV - DV_gt
    loss = huber_loss(
        e_v / std_dev, torch.zeros(e_v.shape, device=y.device), delta=delta
    )

    if with_info:
        # Return sum of squared errors.
        info = {
            "v_se": mse_loss(
                e_v, torch.zeros(e_v.shape, device=y.device), reduction="sum"
            ).item(),
            "v_loss":loss.item()
        }
        return loss, info

    return loss


def position_loss(y, y_gt, with_info=False, delta=1, std_dev=0.1):
    """
    Computes the position loss component between two flattened poses.
    """
    DR = y
    DR_gt, _, _ = unflatten_pose(y_gt)
    e_r = DR - DR_gt
    loss = huber_loss(
        e_r / std_dev, torch.zeros(e_r.shape, device=y.device), delta=delta
    )

    if with_info:
        # Return sum of squared errors.
        info = {
            "r_se": mse_loss(
                e_r, torch.zeros(e_r.shape, device=y.device), reduction="sum"
            ).item(),
            "r_loss":loss.item()
        }
        return loss, info

    return loss

class CustomLoss:
    def __call__(self, y1, y2):
        raise NotImplementedError()

    def write_info(self, writer, epoch, tag):
        pass

class VelocityLoss(CustomLoss):
    def __init__(self):
        self.running_v_loss = 0.0
        self.running_v_se = 0.0
        self.total_samples = 0

    def __call__(self, y1, y2):
        loss, info = velocity_loss(y1, y2, with_info=True)
        self.running_v_loss += info["v_loss"]
        self.running_v_se += info["v_se"]
        self.total_samples += 3 * y1.shape[0]
        return loss

    def write_info(self, writer: SummaryWriter, epoch, tag=""):
        N = self.total_samples
        writer.add_scalar("Loss/Velocity/" + tag, sqrt(self.running_v_loss / N), epoch)
        writer.add_scalar("RMSE/Velocity/" + tag, sqrt(self.running_v_se / N), epoch)
        self.running_v_loss = 0.0
        self.running_v_se = 0.0
        self.total_samples = 0

class PositionLoss(CustomLoss):
    def __init__(self):
        self.running_r_loss = 0.0
        self.running_r_se = 0.0
        self.total_samples = 0

    def __call__(self, y1, y2):
        loss, info = position_loss(y1, y2, with_info=True)
        self.running_r_loss += info["r_loss"]
        self.running_r_se += info["r_se"]
        self.total_samples += 3 * y1.shape[0]
        return loss

    def write_info(self, writer: SummaryWriter, epoch, tag=""):
        N = self.total_samples
        writer.add_scalar("Loss/Position/" + tag, sqrt(self.running_r_loss / N), epoch)
        writer.add_scalar("RMSE/Position/" + tag, sqrt(self.running_r_se / N), epoch)
        self.running_r_loss = 0.0
        self.running_r_se = 0.0
        self.total_samples = 0

class AttitudeLoss(CustomLoss):
    def __init__(self):
        self.running_C_loss = 0.0
        self.running_C_se = 0.0
        self.total_samples = 0

    def __call__(self, y1, y2):
        loss, info = attitude_loss(y1, y2, with_info=True)
        self.running_C_loss += info["C_loss"]
        self.running_C_se += info["C_se"]
        self.total_samples += 3 * y1.shape[0]
        return loss

    def write_info(self, writer: SummaryWriter, epoch, tag=""):
        N = self.total_samples
        writer.add_scalar("Loss/Rotation/" + tag, sqrt(self.running_C_loss / N), epoch)
        writer.add_scalar("RMSE/Rotation/" + tag, sqrt(self.running_C_se / N), epoch)
        self.running_C_loss = 0.0
        self.running_C_se = 0.0
        self.total_samples = 0

class PoseLoss(CustomLoss):
    def __init__(self):
        self.running_C_loss = 0.0
        self.running_v_loss = 0.0
        self.running_r_loss = 0.0
        self.running_C_se = 0.0
        self.running_v_se = 0.0
        self.running_r_se = 0.0
        self.total_samples = 0

    def __call__(self, y1, y2):
        loss, info = pose_loss(y1, y2, with_info=True)
        self.running_C_loss += info["C_loss"]
        self.running_v_loss += info["v_loss"]
        self.running_r_loss += info["r_loss"]
        self.running_C_se += info["C_se"]
        self.running_v_se += info["v_se"]
        self.running_r_se += info["r_se"]
        self.total_samples += 3 * y1.shape[0]
        return loss

    def write_info(self, writer: SummaryWriter, epoch, tag=""):
        N = self.total_samples
        writer.add_scalar("Loss/Position/" + tag, sqrt(self.running_r_loss / N), epoch)
        writer.add_scalar("Loss/Velocity/" + tag, sqrt(self.running_v_loss / N), epoch)
        writer.add_scalar("Loss/Rotation/" + tag, sqrt(self.running_C_loss / N), epoch)
        writer.add_scalar("RMSE/Position/" + tag, sqrt(self.running_r_se / N), epoch)
        writer.add_scalar("RMSE/Velocity/" + tag, sqrt(self.running_v_se / N), epoch)
        writer.add_scalar("RMSE/Rotation/" + tag, sqrt(self.running_C_se / N), epoch)
        self.running_C_loss = 0.0
        self.running_v_loss = 0.0
        self.running_r_loss = 0.0
        self.running_C_se = 0.0
        self.running_v_se = 0.0
        self.running_r_se = 0.0
        self.total_samples = 0

