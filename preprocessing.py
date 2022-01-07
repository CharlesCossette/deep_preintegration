from csv import reader
from torch.utils.data import Dataset
import numpy as np

# TODO: remove bias.

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

def preprocess_data(imu_file, gt_file, output_file):
    imu_data = csv_to_list_of_dict(imu_file)
    gt_data = csv_to_list_of_dict(gt_file)

    f = open(output_file,"w+")
    f.close()
    f = open(output_file,"a")
    
    for imu_meas in imu_data:
        t_imu = imu_meas["#timestamp [ns]"] 
        diff_function = lambda meas : abs(meas["#timestamp"]  - t_imu)
        closest_gt_meas = min(gt_data, key=diff_function)

        processed_meas = preprocess_meas(imu_meas, closest_gt_meas, imu_data[0]["#timestamp [ns]"])
        datastring = ','.join(['%.5f' % num for num in processed_meas])
        datastring += "\n"
        f.write(datastring)


    

def preprocess_meas(imu_meas, gt_meas, time_offset=0.0):

    t = (imu_meas["#timestamp [ns]"] - time_offset) * 1e-9
    
    gyro = np.array([
        imu_meas["w_RS_S_x [rad s^-1]"],
        imu_meas["w_RS_S_y [rad s^-1]"],
        imu_meas["w_RS_S_z [rad s^-1]"],
    ])

    accel = np.array([
        imu_meas["a_RS_S_x [m s^-2]"],
        imu_meas["a_RS_S_y [m s^-2]"],
        imu_meas["a_RS_S_z [m s^-2]"],
    ])/9.80665

    r_zw_a = np.array([
        gt_meas["p_RS_R_x [m]"],
        gt_meas["p_RS_R_y [m]"],
        gt_meas["p_RS_R_z [m]"],
    ])

    v_zw_a = np.array([
        gt_meas["v_RS_R_x [m s^-1]"],
        gt_meas["v_RS_R_y [m s^-1]"],
        gt_meas["v_RS_R_z [m s^-1]"],
    ])

    C_ab = quat_to_matrix(
        gt_meas["q_RS_x []"],
        gt_meas["q_RS_y []"],
        gt_meas["q_RS_z []"],
        gt_meas["q_RS_w []"]
    )
    C_ab = C_ab.flatten()

    b_a = np.array([
        gt_meas["b_a_RS_S_x [m s^-2]"],
        gt_meas["b_a_RS_S_y [m s^-2]"],
        gt_meas["b_a_RS_S_z [m s^-2]"],
    ])

    b_g = np.array([
        gt_meas["b_w_RS_S_x [rad s^-1]"],
        gt_meas["b_w_RS_S_y [rad s^-1]"],
        gt_meas["b_w_RS_S_z [rad s^-1]"],
    ])




    processed = np.hstack((t, gyro - b_g, accel - b_a, r_zw_a, v_zw_a, C_ab))
    return processed

if __name__ == "__main__":
    imu_file = "./data/raw/V1_01_easy/mav0/imu0/data.csv"
    gt_file = "./data/raw/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file,"./data/processed/v1_01_easy.csv")

    imu_file = "./data/raw/V1_02_medium/mav0/imu0/data.csv"
    gt_file = "./data/raw/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file,"./data/processed/v1_02_medium.csv")

    imu_file = "./data/raw/V1_03_difficult/mav0/imu0/data.csv"
    gt_file = "./data/raw/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file,"./data/processed/v1_03_difficult.csv")

    # imu_file = "./data/raw/V2_01_easy/mav0/imu0/data.csv"
    # gt_file = "./data/raw/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    # preprocess_data(imu_file, gt_file,"./data/processed/v2_01_easy.csv")

    # imu_file = "./data/raw/V2_02_medium/mav0/imu0/data.csv"
    # gt_file = "./data/raw/V2_02_medium/mav0/state_groundtruth_estimate0/data.csv"
    # preprocess_data(imu_file, gt_file,"./data/processed/v2_02_medium.csv")

    # imu_file = "./data/raw/V2_03_difficult/mav0/imu0/data.csv"
    # gt_file = "./data/raw/V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv"
    # preprocess_data(imu_file, gt_file,"./data/processed/v2_03_difficult.csv")

    