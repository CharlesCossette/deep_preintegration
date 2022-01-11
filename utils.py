from cv2 import transpose
import numpy as np
from csv import reader
import pandas as pd
import torch
from torch.nn.functional import mse_loss
from pylie import SO3


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
    df = pd.read_csv(filename)
    t = df.iloc[:, 0].to_numpy()
    gyro = df.iloc[:, 1:4].to_numpy().T
    accel = df.iloc[:, 4:7].to_numpy().T
    r_zw_a_gt = df.iloc[:, 7:10].to_numpy().T
    v_zw_a_gt = df.iloc[:, 10:13].to_numpy().T
    C_ab_gt = df.iloc[:, 13:].to_numpy().T
    return {
        "timestamp": t,
        "accel": accel,
        "gyro": gyro,
        "r_zw_a": r_zw_a_gt,
        "v_zwa_a": v_zw_a_gt,
        "C_ab": C_ab_gt,
    }


def LogSO3(C):
    """
    A torch-friendly implementation of the logarithmic map to SO3. This function
    maps a batch of rotation matrices C [N x 3 x 3] to their corresponding
    elements in R^n. Output dimensions are [N x 3]
    """
    dim_batch = C.shape[0]
    Id = torch.eye(3).expand(dim_batch, 3, 3)

    cos_angle = (0.5 * batchtrace(C) - 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    # Clip cos(angle) to its proper domain to avoid NaNs from rounding
    # errors
    angle = cos_angle.acos()
    mask = angle < 1e-8
    if mask.sum() == 0:
        angle = angle.unsqueeze(1).unsqueeze(1)
        return batchvee((0.5 * angle / angle.sin()) * (C - C.transpose(1, 2)))
    elif mask.sum() == dim_batch:
        # If angle is close to zero, use first-order Taylor expansion
        return batchvee(C - Id)
    phi = batchvee(C - Id)
    angle = angle
    phi[~mask] = batchvee(
        (0.5 * angle[~mask] / angle[~mask].sin()).unsqueeze(1).unsqueeze(2)
        * (C[~mask] - C[~mask].transpose(1, 2))
    )
    return phi


def batchvee(Phi):
    return torch.stack((Phi[:, 2, 1], Phi[:, 0, 2], Phi[:, 1, 0]), dim=1)


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