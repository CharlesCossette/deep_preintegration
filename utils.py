from cv2 import transpose
import numpy as np
from csv import reader
import pandas as pd
import torch
from torch.nn.functional import mse_loss
from pylie.torch import SO3


def quat_to_matrix(x, y, z, w):
    """
    Takes a quaternion and converts it to a 3x3 numpy array rotation matrix/DCM.

    Formula taken from the EuRoC dataset paper.
    """

    # eps = np.array([x, y, z]).reshape((-1, 1))
    q_bar = np.array([x, y, z]).reshape((-1, 1))
    # eta = w
    # return (
    #     (1 - 2 * np.matmul(np.transpose(eps), eps)) * np.eye(3)
    #     + 2 * np.matmul(eps, np.transpose(eps))
    #     - 2 * eta * cross_matrix(eps)
    # )
    q_bar_cross = cross_matrix(q_bar)
    return (
        (w ** 2) * np.eye(3)
        + 2 * w * q_bar_cross
        + np.matmul(q_bar_cross, q_bar_cross)
        + np.matmul(q_bar, np.transpose(q_bar))
    )


def cross_matrix(v):
    v = v.flatten()
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def csv_to_list_of_dict(filename):
    data = []
    with open(filename) as f:
        r = reader(f)
        headers = next(r)
        for row in r:
            data.append({key.strip(): float(value) for key, value in zip(headers, row)})
    return data


def load_processed_data(filename):
    # TODO: get rid of this function. merge with RmiDataset somehow.
    data = torch.Tensor(pd.read_csv(filename).values)
    t = data[:, 0]
    gyro = data[:, 1:4].T
    accel = data[:, 4:7].T
    r_zw_a_gt = data[:, 7:10].T
    v_zw_a_gt = data[:, 10:13].T
    C_ab_gt = data[:, 13:].T
    return {
        "timestamp": t,
        "accel": accel,
        "gyro": gyro,
        "r_zw_a": r_zw_a_gt,
        "v_zwa_a": v_zw_a_gt,
        "C_ab": C_ab_gt,
    }


def batchtrace(mat):
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    tr = (torch.eye(mat.shape[1], dtype=mat.dtype) * mat).sum(dim=1).sum(dim=1)

    return tr.view(mat.shape[0])


def trace_loss(C1, C2):
    """
    An error metric between two DCMs based on trace:

    J = trace(eye - C)

    where C = C1 * C2.T
    """
    dim_batch = C1.shape[0]
    Id = torch.eye(3).expand(dim_batch, 3, 3)
    DC = torch.matmul(C1, torch.transpose(C2, 1, 2))
    return torch.sum(batchtrace(Id - DC)) / dim_batch


def imu_dead_reckoning(t, r0, v0, C0, gyro, accel):
    r = r0
    v = v0
    C = C0
    g = torch.Tensor([0, 0, -9.80665]).reshape((-1, 1))
    t_data = [torch.Tensor([0.0])]
    r_data = [r]
    v_data = [v]
    C_data = [C]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        w = gyro[:, i - 1].reshape((-1, 1))
        a = accel[:, i - 1].reshape((-1, 1))
        r = r + v * dt + 0.5 * g * (dt ** 2) + 0.5 * C @ a * (dt ** 2)
        v = v + g * dt + C @ a * dt
        C = C @ SO3.Exp(dt * w).squeeze()

        t_data.append(t[i])
        r_data.append(r)
        v_data.append(v)
        C_data.append(C)

    t_data = torch.hstack(t_data)
    r_data = torch.hstack(r_data)
    v_data = torch.hstack(v_data)
    return {"t": t_data, "r": r_data, "v": v_data, "C": C_data}


def unflatten_pose(x):
    """
    Decomposes an [N x 15] array into position, velocity, and rotation arrays.

    """

    is_batch = len(x.shape) > 1

    if not is_batch == 1:
        C = x[0:9].view((3, 3))
        v = x[9:12].view((3, 1))
        r = x[12:].view((3, 1))
    else:
        dim_batch = x.shape[0]
        C = x[:, 0:9].view((dim_batch, 3, 3))
        v = x[:, 9:12].view((dim_batch, 3))
        r = x[:, 12:].view((dim_batch, 3))

    return r, v, C


def flatten_pose(r, v, C):
    """
    Takes [3 x 1] position, [3 x 1] velocity, and [3 x 3] rotation arrays and
    flattens + stacks them all into a [1 x 15] array.

    TODO: accept batch
    """

    if isinstance(r, np.ndarray):
        return np.hstack((C.flatten(), v.flatten(), r.flatten()))
    elif isinstance(r, torch.Tensor):
        return torch.hstack((C.flatten(), v.flatten(), r.flatten()))
    else:
        raise RuntimeError("Not an accepted variable type.")


def count_parameters(model):
    """
    Counts the number of trainable parameters in a neural network.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
