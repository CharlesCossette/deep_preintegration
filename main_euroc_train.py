import torch
from model import RmiModel, RmiNet2
from dataset import DeltaRmiDataset, RmiDataset
from torch.utils.data import Subset, ConcatDataset, DataLoader
from training import train, valid_loop
from losses import VelocityLoss, PoseLoss
from utils import unflatten_pose
from pylie.torch import SO3
import matplotlib.pyplot as plt

# TODO: estimate covariance of RMIs
# TODO: estimate scaling of RMIs

N = 1000  # Window size
stride = 50
batch_size = 256
epochs = 2000
load_file = None
#load_file = "rminet2.pt"
calib_file = "./results/best_calibration_saved.pt"
lr = 0.01
weight_decay = 0
output_window = 400
use_gpu = True
#torch.set_default_dtype(torch.float64)

# ##############################################################################
# Divide dataset into training and validation: take first 50 seconds as training.

raw = [
    RmiDataset("./data/processed/v1_03_difficult.csv", N, stride, output_window=output_window),
    RmiDataset("./data/processed/v2_01_easy.csv", N, stride, output_window=output_window),
    RmiDataset("./data/processed/v2_02_medium.csv", N, stride, output_window=output_window),
    RmiDataset("./data/processed/v2_03_difficult.csv", N, stride, output_window=output_window),
    RmiDataset("./data/processed/mh_02_easy.csv", N, stride, output_window=output_window),
    RmiDataset("./data/processed/mh_03_medium.csv", N, stride, output_window=output_window),
    RmiDataset("./data/processed/mh_04_difficult.csv", N, stride, output_window=output_window),
]

# trainset_list = []
# validset_list = []
# for dataset in raw:
#     idx = dataset.get_index_of_time(80)
#     trainset_list.append(Subset(dataset, list(range(idx))))
#     validset_list.append(Subset(dataset, list(range(idx, len(dataset)))))

trainset = ConcatDataset([raw[0], raw[1], raw[3], raw[4], raw[6]])
validset = ConcatDataset([raw[2], raw[5]])

calib_results = torch.load(calib_file, map_location="cpu")
calibrator = RmiModel(output_window = output_window)
calibrator.load_state_dict(calib_results)

delta_trainset = DeltaRmiDataset(trainset, calibrator)
delta_validset = DeltaRmiDataset(validset, calibrator)

# #############################################################################
# Get output standard deviation
x, y = next(iter(DataLoader(delta_trainset, batch_size = len(delta_trainset))))
xv, yv = next(iter(DataLoader(delta_validset, batch_size = len(delta_validset))))

x_mean = torch.mean(x[:,])
y_std_dev = torch.std(y, dim=0, unbiased=True)
print(y_std_dev)
# #############################################################################


compare_loader = DataLoader(delta_validset, batch_size=batch_size)
compare_loss = valid_loop(lambda x: torch.zeros(x.shape[0],3), compare_loader, VelocityLoss())
print(compare_loss)

""" RMINet2"""
net = RmiNet2(output_std_dev=y_std_dev[3:6])
net.set_calibration(calib_results["calib_mat"], calib_results["bias"], False)
loss_fn = VelocityLoss()
output_file = "rminet2.pt"


net_eval = RmiNet2(output_std_dev=y_std_dev[3:6])
net_eval.set_calibration(calib_results["calib_mat"], calib_results["bias"], False)
net_eval.load_state_dict(torch.load("./results/best_rminet2.pt" , map_location="cpu"))
net_eval.eval()
y_hat = net_eval(x).detach().numpy()
yv_hat = net_eval(xv).detach().numpy()


fig, axs = plt.subplots(3,1)
axs[0].plot(y[:,3],label="Model Error")
axs[0].plot(y_hat[:,0],label="NN")
axs[1].plot(y[:,4])
axs[1].plot(y_hat[:,1])
axs[2].plot(y[:,5])
axs[2].plot(y_hat[:,2])
axs[0].legend()
axs[0].set_title("Velocity Training Error")
axs[2].set_xlabel("Sample Number")

fig, axs = plt.subplots(3,1)
axs[0].plot(yv[:,3],label="Model Error")
axs[0].plot(yv_hat[:,0],label="NN")
axs[1].plot(yv[:,4])
axs[1].plot(yv_hat[:,1])
axs[2].plot(yv[:,5])
axs[2].plot(yv_hat[:,2])
axs[0].legend()
axs[0].set_title("Velocity Validation Error")
axs[2].set_xlabel("Sample Number")
plt.show()


train(
    net,
    trainset=delta_trainset,
    batch_size=batch_size,
    epochs=epochs,
    validset=delta_validset,
    output_file=output_file,
    weights_file=load_file,
    lr=lr,
    loss_fn=loss_fn,
    use_gpu=use_gpu,
    weight_decay=weight_decay
)
# # print(" ANALYTICAL MODEL LOSS: " + str(model_loss))

print("done")