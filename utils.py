
import numpy as np
from csv import reader
import pandas as pd
import torch

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
    phi = torch.acos((batchtrace(C) - 1)/2)
    b = 1/(2*torch.sin(phi))
    c = torch.Tensor([
            C[:,2,1] - C[:,1,2],
            C[:,0,2] - C[:,2,0],
            C[:,1,0] - C[:,0,1]
            ])
    return torch.matmul(phi, torch.matmul(b, c))

def batchtrace(M):
    M.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)