
from cv2 import transpose
import numpy as np
from csv import reader
import pandas as pd
import torch
from torch.nn.functional import mse_loss

def quat_to_matrix(x,y,z,w):
    """
    Takes a quaternion message from geometry_msgs/Quaternion and converts it
    to a 3x3 numpy array rotation matrix/DCM.
    """

    eps = np.array([x, y, z]).reshape((-1, 1))

    eta = w
    return (
        (1 - 2 * np.matmul(np.transpose(eps), eps)) * np.eye(3)
        + 2 * np.matmul(eps, np.transpose(eps))
        - 2 * eta * cross_matrix(eps)
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
    accel = df.iloc[:, 1:4].to_numpy()
    gyro = df.iloc[:, 4:7].to_numpy()
    r_zw_a_gt = df.iloc[:, 7:10].to_numpy()
    v_zw_a_gt = df.iloc[:, 10:13].to_numpy()
    C_ab_gt = df.iloc[:, 13:].to_numpy()
    return {
        "timestamp":t,
        "accel":accel,
        "gyro":gyro,
        "r_zw_a":r_zw_a_gt,
        "v_zwa_a":v_zw_a_gt,
        "C_ab":C_ab_gt
        }

def LogSO3(C):
    """
    A torch-friendly implementation of the logarithmic map to SO3. This function 
    maps a batch of rotation matrices C [N x 3 x 3] to their corresponding 
    elements in R^n. Output dimensions are [N x 3]
    """
    dim_batch = C.shape[0]
    Id = torch.eye(3).expand(dim_batch, 3, 3)

    cos_angle = (0.5 * batchtrace(C) - 0.5).clamp(-1. + 1e-7, 1. - 1e-7)
    # Clip cos(angle) to its proper domain to avoid NaNs from rounding
    # errors
    angle = cos_angle.acos()
    mask = angle < 1e-8
    if mask.sum() == 0:
        angle = angle.unsqueeze(1).unsqueeze(1)
        return batchvee((0.5 * angle/angle.sin())*(C - C.transpose(1, 2)))
    elif mask.sum() == dim_batch:
        # If angle is close to zero, use first-order Taylor expansion
        return batchvee(C - Id)
    phi = batchvee(C - Id)
    angle = angle
    phi[~mask] = batchvee((0.5 * angle[~mask]/angle[~mask].sin()).unsqueeze(
        1).unsqueeze(2)*(C[~mask] - C[~mask].transpose(1, 2)))
    return phi


def batchvee(Phi):
    return torch.stack((Phi[:, 2, 1],
                        Phi[:, 0, 2],
                        Phi[:, 1, 0]), dim=1)
def batchtrace(mat):
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    tr =  (torch.eye(mat.shape[1], dtype=mat.dtype) * mat).sum(dim=1).sum(dim=1)
    
    return tr.view(mat.shape[0])

def trace_loss(C1, C2):
    dim_batch = C1.shape[0]
    Id = torch.eye(3).expand(dim_batch, 3, 3)
    DC = torch.matmul(C1, torch.transpose(C2, 1,2))
    return torch.sum(batchtrace(Id - DC))/dim_batch

def pose_loss(y, y_gt):
    DC = torch.reshape(y[:,0:9],(y.shape[0],3,3))
    DC_gt = torch.reshape(y_gt[:,0:9],(y_gt.shape[0],3,3))
    phi = LogSO3(torch.matmul(DC, torch.transpose(DC_gt, 1,2)))
    e = torch.hstack((phi, y[:,9:] - y_gt[:,9:]))
    if torch.any(torch.isnan(e)):
        raise RuntimeError("Nan for some reason.")
    return mse_loss(e, torch.zeros(e.shape))
    return trace_loss(DC, DC_gt) + mse_loss(y[:,9:], y_gt[:,9:])