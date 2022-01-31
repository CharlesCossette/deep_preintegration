import torch

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
        self.total_samples += 3 * y1.shape[0]
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

class DeltaTransRmiNet(BaseNet):
    """
    Convolutional neural network based RMI corrector. The network outputs a
    "delta" which is then "added" to the analytical RMIs to produce the final
    estimate.
    """

    def __init__(self, window_size=200):
        self._window_size = window_size
        super(DeltaTransRmiNet, self).__init__()

        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                6,
                16,
                7,
                padding="same",
                padding_mode="replicate",
                dilation=1,
                bias=False,
            ),
            torch.nn.BatchNorm1d(16),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            torch.nn.Conv1d(
                16,
                32,
                7,
                padding="same",
                padding_mode="replicate",
                dilation=16,
                bias=False,
            ),
            torch.nn.BatchNorm1d(32),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            torch.nn.Conv1d(
                32,
                64,
                7,
                padding="same",
                padding_mode="replicate",
                dilation=64,
                bias=False,
            ),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            # torch.nn.Conv1d(
            #     64,
            #     128,
            #     7,
            #     padding="same",
            #     padding_mode="replicate",
            #     dilation=64,
            #     bias=False,
            # ),
            # torch.nn.BatchNorm1d(128),
            # torch.nn.GELU(),
            # torch.nn.Dropout(0.2),
            torch.nn.Conv1d(
                64, 6, 7, padding="same", padding_mode="replicate", dilation=1
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
        )
        self.linear_layer = torch.nn.Sequential(
            # torch.nn.LazyLinear(30),
            # torch.nn.GELU(),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(6, 6),
        )
        self.calib_mat = torch.nn.Parameter(0.05 * torch.randn((6, 6)))

    def forward(self, x: torch.Tensor):
        x = x[:, 1:, :]
        # x = self.norm(x)
        M = (self.I + self.calib_mat).expand((x.shape[0], 6, 6))
        x = torch.matmul(M, x)
        x = self.conv_layer(x)
        x = torch.mean(x, dim=2)
        x = self.linear_layer(x)
        return x


class DeltaRotRmiNet(torch.nn.Module):
    """
    Convolutional neural network based RMI corrector. The network outputs a
    "delta" which is then "added" to the analytical RMIs to produce the final
    estimate.
    """

    def __init__(self, window_size=200):
        self._window_size = window_size
        super(DeltaRotRmiNet, self).__init__()
        in_dim = 6
        out_dim = 3
        dropout = 0
        momentum = 0.1
        # channel dimension
        c0 = 16
        c1 = 2 * c0
        c2 = 2 * c1
        c3 = 2 * c2
        # kernel dimension (odd number)
        k0 = 7
        k1 = 7
        k2 = 7
        k3 = 7
        # dilation dimension
        d0 = 4
        d1 = 4
        d2 = 4
        # padding
        p0 = (k0 - 1) + d0 * (k1 - 1) + d0 * d1 * (k2 - 1) + d0 * d1 * d2 * (k3 - 1)
        # nets
        self.conv_layer = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((p0, 0)),  # padding at start
            torch.nn.Conv1d(in_dim, c0, k0, dilation=1),
            torch.nn.BatchNorm1d(c0, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c0, c1, k1, dilation=d0),
            torch.nn.BatchNorm1d(c1, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c1, c2, k2, dilation=d0 * d1),
            torch.nn.BatchNorm1d(c2, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c2, c3, k3, dilation=d0 * d1 * d2),
            torch.nn.BatchNorm1d(c3, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c3, out_dim, 1, dilation=1),
            torch.nn.Flatten(),  # no padding at end
        )
        self.linear_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(3),
            # torch.nn.GELU(),
            # torch.nn.Linear(50, 3),
        )

    def forward(self, x: torch.Tensor):
        x = x.detach().clone()
        x = self.conv_layer(x)
        x = self.linear_layer(x)
        return x