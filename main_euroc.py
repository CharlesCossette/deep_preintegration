from math import pi
from model import DeltaTransRmiNet, DeltaRotRmiNet
import matplotlib.pyplot as plt
from utils import load_processed_data, bmtm
import seaborn as sns
import torch
from evaluation import seperated_nets_test, imu_dead_reckoning, trans_net_violin
from pylie.torch import SO3
import numpy as np

sns.set_theme(context="paper")

N = 500  # window size
# torch.set_default_dtype(torch.float64)
# test_file = "./data/processed/v1_03_difficult.csv"
test_file = "./data/processed/v1_01_easy.csv"
# data = load_processed_data(test_file)
# t = data["timestamp"]
# r_gt = data["r_zw_a"]
# v_gt = data["v_zwa_a"]
# C_gt = data["C_ab"]
# accel = data["accel"]
# gyro = data["gyro"]

# # Do classical dead reckoning
# traj = imu_dead_reckoning(test_file)


trans_net = DeltaTransRmiNet(N)
trans_net.load_state_dict(torch.load("./results/best_dtransrminet_weights.pt"))
# rot_net = DeltaRotRmiNet(N)
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

data = trans_net_violin(trans_net, test_file)
fig, ax = plt.subplots()
ax.violinplot(data)
fig = plt.figure()
ax = plt.plot(data.T)
# %%
plt.show()
