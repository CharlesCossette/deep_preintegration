#%%
from math import pi
from model import  RmiModel, RmiNet2
import matplotlib.pyplot as plt
from utils import load_processed_data, bmtm
import seaborn as sns
import torch
from evaluation import (
    rmi_estimator_test,
    seperated_nets_test,
    imu_dead_reckoning,
    trans_net_violin,
)
from pylie.torch import SO3
import numpy as np

sns.set_theme(context="paper")

N = 1000  # window size
# torch.set_default_dtype(torch.float64)
#test_file = "./data/processed/v1_03_difficult.csv"  # Training set

#test_file = "./data/processed/v1_01_easy.csv"
# test_file = "./data/processed/v2_02_medium.csv"
test_file = "./data/processed/mh_01_easy.csv"
# test_file = "./data/processed/mh_05_difficult.csv"
# data = load_processed_data(test_file)
# t = data["timestamp"]
# r_gt = data["r_zw_a"]
# v_gt = data["v_zwa_a"]
# C_gt = data["C_ab"]
# accel = data["accel"]
# gyro = data["gyro"]

# # Do classical dead reckoning
# traj = imu_dead_reckoning(test_file)


# trans_net = DeltaTransRmiNet(N)
# trans_net.load_state_dict(torch.load("./results/best_dtransrminet_weights.pt"))
# # rot_net = DeltaRotRmiNet(N)
# rot_net.load_state_dict(torch.load("./results/drotrminet_weights.pt"))
# traj_net = seperated_nets_test(trans_net, rot_net, test_file)

# fig, axs = plt.subplots(3, 1)
# fig.suptitle("Position Trajectory")
# label = ["x", "y", "z"]
# for i in range(3):
#     axs[i].plot(t, r_gt[i, :], label="Ground Truth")
#     axs[i].plot(traj["t"], traj["r"][i, :], label="Traditional")
#     axs[i].plot(traj_net["t"], traj_net["r"][i, :], label="RMI-Net")
#     axs[i].set_ylabel(label[i])

# axs[-1].set_xlabel("Time (s)")
# axs[0].legend()

# fig, axs = plt.subplots(3, 1)
# fig.suptitle("Velocity Trajectory")
# label = ["x", "y", "z"]
# for i in range(3):
#     axs[i].plot(t, v_gt[i, :], label="Ground Truth")
#     axs[i].plot(traj["t"], traj["v"][i, :], label="Traditional")
#     axs[i].plot(traj_net["t"], traj_net["v"][i, :], label="RMI-Net")
#     axs[i].set_ylabel(label[i])

# axs[-1].set_xlabel("Time (s)")
# axs[0].legend()

# fig, axs = plt.subplots(3, 1)
# fig.suptitle("Attitude Error (degrees)")
# label = ["x", "y", "z"]
# e_att_net = SO3.Log(
#     torch.matmul(torch.transpose(traj_net["C"], 1, 2), traj_net["C_gt"])
# )
# e_att = SO3.Log(torch.matmul(torch.transpose(traj["C"], 1, 2), C_gt))
# for i in range(3):
#     axs[i].plot(traj["t"], e_att[:, i] * (360 / (2 * pi)), label="Traditional")
#     axs[i].plot(traj_net["t"], e_att_net[:, i] * (360 / (2 * pi)), label="RMI-Net")
#     axs[i].set_ylabel(label[i])

# ax = plt.plot(data.T)
# axs[-1].set_xlabel("Time (s)")
# axs[0].legend()

net = RmiModel(N)
net.load_state_dict(torch.load("./results/best_calibration_saved.pt", map_location="cpu"))
data = rmi_estimator_test(net, test_file, N)

# %%
net_bias = RmiModel(N)
net_bias.set_calibration(torch.zeros((6,6)), -torch.Tensor([-0.002229,0.0207,0.07635,-0.012492,0.547666,0.069073]))
data_bias = rmi_estimator_test(net_bias, test_file, N)

net2 = RmiNet2(N)
net2.load_state_dict(torch.load("./results/best_rminet2.pt", map_location="cpu"))
data_net = rmi_estimator_test(net2, test_file, N)

#%%
fig, ax = plt.subplots(1, 3)
temp = torch.vstack((data[0:2,:], data_net[0,:],data[2:4,:], data_net[2,:],data[4:6,:], data_net[4,:],))
ax[0] = sns.violinplot(data=data[0:3, :].T, ax=ax[0], cut=0)
ax[0].set_xticklabels(["Calibrated", "Raw"])
ax[0].set_title("Position RMSE [m]")
ax[1] = sns.violinplot(data=data[3:6, :].T, ax=ax[1], cut=0)
ax[1].set_xticklabels(["Calibrated", "Raw"])
ax[1].set_title("Velocity RMSE [m/s]")
ax[2] = sns.violinplot(data=data[6:9, :].T, ax=ax[2], cut=0)
ax[2].set_xticklabels(["Calibrated", "Raw"])
ax[2].set_title("Attitude RMSE [rad]")

temp = data.clone()
temp[[1, 3, 5], :] = data_bias[[0, 2, 4], :]

fig, ax = plt.subplots(1, 3)
ax[0] = sns.violinplot(data=temp[0:2, :].T, ax=ax[0], cut=0)
ax[0].set_xticklabels(["Calibrated", "Bias Removed"])
ax[0].set_title("Position RMSE [m]")
ax[1] = sns.violinplot(data=temp[2:4, :].T, ax=ax[1], cut=0)
ax[1].set_xticklabels(["Calibrated", "Bias Removed"])
ax[1].set_title("Velocity RMSE [m/s]")
ax[2] = sns.violinplot(data=temp[4:6, :].T, ax=ax[2], cut=0)
ax[2].set_xticklabels(["Calibrated", "Bias Removed"])
ax[2].set_title("Attitude RMSE [rad]")


# #%%
fig, ax = plt.subplots(3, 1)
ax[0].plot(data[0, :].T, label="Calibrated")
ax[0].plot(data[1, :].T, label="Raw")
ax[0].set_ylabel("Pos. [m]")
ax[1].plot(data[2, :].T)
ax[1].plot(data[3, :].T)
ax[1].set_ylabel("Vel. [m/s]")
ax[2].plot(data[4, :].T)
ax[2].plot(data[5, :].T)
ax[2].set_xlabel("Window number")
ax[2].set_ylabel("Att. [rad]")
ax[0].legend()
fig.suptitle("RMSE throughout test dataset")

fig, ax = plt.subplots(3, 1)
ax[0].plot(temp[0, :].T, label="Calibrated")
ax[0].plot(temp[1, :].T, label="Bias Removed")
ax[0].set_ylabel("Pos. [m]")
ax[1].plot(temp[2, :].T)
ax[1].plot(temp[3, :].T)
ax[1].set_ylabel("Vel. [m/s]")
ax[2].plot(temp[4, :].T)
ax[2].plot(temp[5, :].T)
ax[2].set_xlabel("Window number")
ax[2].set_ylabel("Att. [rad]")
ax[0].legend()
fig.suptitle("RMSE throughout test dataset")

# data = data2
# fig, ax = plt.subplots(3, 1)
# ax[0].plot(data[0, :].T, label="Calibrated")
# ax[0].plot(data[1, :].T, label="Raw")
# ax[1].plot(data[2, :].T)
# ax[1].plot(data[3, :].T)
# ax[2].plot(data[4, :].T)
# ax[2].plot(data[5, :].T)
# ax[0].legend()
plt.show()

# %%
