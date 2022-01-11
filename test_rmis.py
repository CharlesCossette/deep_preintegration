from tkinter import X
from model import *
from utils import imu_dead_reckoning

data = RmiDataset("./results/v1_01_easy.csv")
x, rmis, poses = data.get_item_with_poses(0)

y = get_rmis(x)

t = x[0,:].detach().to("cpu").numpy()
gyro = x[1:4,:].detach().to("cpu").numpy()
accel = x[4:7,:].detach().to("cpu").numpy()

traj = imu_dead_reckoning(t, poses[0][0], poses[0][1], poses[0][2], gyro, accel)
