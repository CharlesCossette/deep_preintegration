import numpy as np
import pandas as pd
from utils import *
from math import floor


def preprocess_data(imu_file, gt_file, output_file):
    print("Processing the following files:")
    print(imu_file)
    print(gt_file)

    imu_data = pd.read_csv(imu_file)
    gt_data = pd.read_csv(gt_file)
    imu_data.columns = imu_data.columns.str.strip()
    gt_data.columns = gt_data.columns.str.strip()

    df = pd.merge_asof(
        imu_data,
        gt_data,
        left_on="#timestamp [ns]",
        right_on="#timestamp",
        direction="nearest",
    )
    df = df.rename(
        columns={
            "#timestamp [ns]": "t",
            "w_RS_S_x [rad s^-1]": "wx",
            "w_RS_S_y [rad s^-1]": "wy",
            "w_RS_S_z [rad s^-1]": "wz",
            "a_RS_S_x [m s^-2]": "ax",
            "a_RS_S_y [m s^-2]": "ay",
            "a_RS_S_z [m s^-2]": "az",
            "p_RS_R_x [m]": "px",
            "p_RS_R_y [m]": "py",
            "p_RS_R_z [m]": "pz",
            "v_RS_R_x [m s^-1]": "vx",
            "v_RS_R_y [m s^-1]": "vy",
            "v_RS_R_z [m s^-1]": "vz",
            "b_a_RS_S_x [m s^-2]": "bax",
            "b_a_RS_S_y [m s^-2]": "bay",
            "b_a_RS_S_z [m s^-2]": "baz",
            "b_w_RS_S_x [rad s^-1]": "bwx",
            "b_w_RS_S_y [rad s^-1]": "bwy",
            "b_w_RS_S_z [rad s^-1]": "bwz",
            "q_RS_x []": "qx",
            "q_RS_y []": "qy",
            "q_RS_z []": "qz",
            "q_RS_w []": "qw",
        }
    )

    df["t"] = df["t"] - df.loc[0, "t"]
    df["t"] *= 1e-9

    dcm_data = []
    column_names = ["c11", "c12", "c13", "c21", "c22", "c23", "c31", "c32", "c33"]
    for _, row in df.iterrows():
        C_ab = quat_to_matrix(
            row["qx"],
            row["qy"],
            row["qz"],
            row["qw"],
        )
        C_ab = C_ab.flatten()
        dcm_data.append(C_ab)
    df_dcm = pd.DataFrame(dcm_data, columns=column_names)
    df = pd.concat([df, df_dcm], axis=1)
    df.to_csv(output_file, index=None)


if __name__ == "__main__":
    imu_file = "./data/raw/V1_01_easy/mav0/imu0/data.csv"
    # gt_file = "./data/raw/V1_01_easy/V1_01_easy_corrected_groundtruth.csv"
    gt_file = "./data/raw/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/v1_01_easy.csv")

    imu_file = "./data/raw/V1_02_medium/mav0/imu0/data.csv"
    gt_file = "./data/raw/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/v1_02_medium.csv")

    imu_file = "./data/raw/V1_03_difficult/mav0/imu0/data.csv"
    gt_file = "./data/raw/V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/v1_03_difficult.csv")

    imu_file = "./data/raw/V2_01_easy/mav0/imu0/data.csv"
    gt_file = "./data/raw/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/v2_01_easy.csv")

    imu_file = "./data/raw/V2_02_medium/mav0/imu0/data.csv"
    gt_file = "./data/raw/V2_02_medium/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/v2_02_medium.csv")

    imu_file = "./data/raw/V2_03_difficult/mav0/imu0/data.csv"
    gt_file = "./data/raw/V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/v2_03_difficult.csv")

    imu_file = "./data/raw/MH_01_easy/mav0/imu0/data.csv"
    gt_file = "./data/raw/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/mh_01_easy.csv")

    imu_file = "./data/raw/MH_02_easy/mav0/imu0/data.csv"
    gt_file = "./data/raw/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/mh_02_easy.csv")

    imu_file = "./data/raw/MH_03_medium/mav0/imu0/data.csv"
    gt_file = "./data/raw/MH_03_medium/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/mh_03_medium.csv")

    imu_file = "./data/raw/MH_04_difficult/mav0/imu0/data.csv"
    gt_file = "./data/raw/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/mh_04_difficult.csv")

    imu_file = "./data/raw/MH_05_difficult/mav0/imu0/data.csv"
    gt_file = "./data/raw/MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv"
    preprocess_data(imu_file, gt_file, "./data/processed/mh_05_difficult.csv")
