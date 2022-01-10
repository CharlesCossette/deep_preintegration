from model import *
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

data = load_processed_data("./data/processed/v1_01_easy.csv")
t = data["timestamp"]
r_zw_a_gt = data["r_zw_a"]
v_zwa_a_gt = data["v_zwa_a"]
C_ab_gt = data["C_ab"]
g_a = np.array([0, 0, -9.80665]).reshape((-1, 1))


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(r_zw_a_gt[:,0],r_zw_a_gt[:,1],r_zw_a_gt[:,2])
# ax.set_xlim(-5,5)
# ax.set_ylim(-5,5)
# ax.set_zlim(0,10)

with torch.no_grad():
    net = RmiNet(window_size=200)
    net.eval()
    net.load_state_dict(torch.load("./results/rminet_weights2.pth"))
    r_j = r_zw_a_gt[0,:].reshape((-1,1))
    v_j = v_zwa_a_gt[0,:].reshape((-1,1))
    C_j = C_ab_gt[0,:].reshape((3,3))

    r_net = [r_j]
    for i in range(200,r_zw_a_gt.shape[0],200):
        gyro_window = data["gyro"][i-200:i,:]
        accel_window = data["accel"][i-200:i,:]
        imu_window = torch.from_numpy(np.vstack((gyro_window.T, accel_window.T)))
        imu_window = torch.reshape(imu_window, (1,imu_window.shape[0], imu_window.shape[1]))
        imu_window = imu_window.to(torch.float32)
        rmis = net(imu_window)
        DC = torch.reshape(rmis[0,0:9],(3,3)).numpy()
        DV = torch.reshape(rmis[0,9:12],(3,1)).numpy()
        DR= torch.reshape(rmis[0,12:15],(3,1)).numpy()
        DT = t[i] - t[i-200]
        r_i = r_j 
        v_i = v_j
        C_i = C_j
        C_j = C_i @ DC
        v_j = v_i + g_a * DT + C_i @ DV
        r_j = r_i + v_i * DT + 0.5 * g_a * DT ** 2 + C_i @ DR
        r_net.append(r_j)
 
r_net = np.hstack(r_net)
ax.plot(r_net[0,:], r_net[1,:], r_net[2,:])
plt.show()
