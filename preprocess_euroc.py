import numpy as np
import pandas as pd
from utils import *


def preprocess_data(imu_file, gt_file, output_file):
    imu_data = pd.read_csv(imu_file)
    gt_data = pd.read_csv(gt_file)
    imu_data.columns = imu_data.columns.str.strip()
    gt_data.columns = gt_data.columns.str.strip()

    f = open(output_file,"w+")
    f.close()
    f = open(output_file,"a")
    
    
    for _, imu_meas in imu_data.iterrows():
        t_imu = imu_meas["#timestamp [ns]"]
        result_index = gt_data['#timestamp'].sub(t_imu).abs().idxmin()
        closest_gt_meas = gt_data.loc[result_index,:]
        processed_meas = preprocess_meas(imu_meas, closest_gt_meas, imu_data["#timestamp [ns]"][0])
        datastring = ','.join(['%.5f' % num for num in processed_meas])
        datastring += "\n"
        f.write(datastring)    

def preprocess_meas(imu_meas, gt_meas, time_offset=0.0):

    t = (imu_meas["#timestamp [ns]"] - time_offset) * 1e-9

    gyro = imu_meas[[
        "w_RS_S_x [rad s^-1]",
        "w_RS_S_y [rad s^-1]",
        "w_RS_S_z [rad s^-1]"
        ]].to_numpy()
   
    accel = imu_meas[[
        "a_RS_S_x [m s^-2]",
        "a_RS_S_y [m s^-2]",
        "a_RS_S_z [m s^-2]"
        ]].to_numpy()/9.80665

    r_zw_a = gt_meas[[
        "p_RS_R_x [m]",
        "p_RS_R_y [m]",
        "p_RS_R_z [m]"
        ]].to_numpy()

    v_zw_a = gt_meas[[
        "v_RS_R_x [m s^-1]",
        "v_RS_R_y [m s^-1]",
        "v_RS_R_z [m s^-1]"
        ]].to_numpy()

    C_ab = quat_to_matrix(
        gt_meas["q_RS_x []"],
        gt_meas["q_RS_y []"],
        gt_meas["q_RS_z []"],
        gt_meas["q_RS_w []"]
    )
    C_ab = C_ab.flatten()

    b_a = gt_meas[[
        "b_a_RS_S_x [m s^-2]",
        "b_a_RS_S_y [m s^-2]",
        "b_a_RS_S_z [m s^-2]"
        ]].to_numpy()

    b_g = gt_meas[[
        "b_w_RS_S_x [rad s^-1]",
        "b_w_RS_S_y [rad s^-1]",
        "b_w_RS_S_z [rad s^-1]"
        ]].to_numpy()

    processed = np.hstack((t, gyro - b_g, accel - b_a, r_zw_a, v_zw_a, C_ab))
    return processed

if __name__ == "__main__":
    imu_file = "./data/raw/V1_01_easy/mav0/imu0/data.csv"
    gt_file = "./data/raw/V1_01_easy/V1_01_easy_corrected_groundtruth.csv"
    #gt_file = "./data/raw/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file,"./data/processed/v1_01_easy.csv")

    # imu_file = "./data/raw/V1_02_medium/mav0/imu0/data.csv"
    # gt_file = "./data/raw/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv"
    # preprocess_data(imu_file, gt_file,"./data/processed/v1_02_medium.csv")

    # imu_file = "./data/raw/V1_03_difficult/mav0/imu0/data.csv"
    # gt_file = "./data/raw/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv"
    # preprocess_data(imu_file, gt_file,"./data/processed/v1_03_difficult.csv")

    # imu_file = "./data/raw/V2_01_easy/mav0/imu0/data.csv"
    # gt_file = "./data/raw/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    # preprocess_data(imu_file, gt_file,"./data/processed/v2_01_easy.csv")

    # imu_file = "./data/raw/V2_02_medium/mav0/imu0/data.csv"
    # gt_file = "./data/raw/V2_02_medium/mav0/state_groundtruth_estimate0/data.csv"
    # preprocess_data(imu_file, gt_file,"./data/processed/v2_02_medium.csv")

    # imu_file = "./data/raw/V2_03_difficult/mav0/imu0/data.csv"
    # gt_file = "./data/raw/V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv"
    # preprocess_data(imu_file, gt_file,"./data/processed/v2_03_difficult.csv")

    