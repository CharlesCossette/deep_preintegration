import torch
from model import RmiModel, RmiNet2
from dataset import RmiDataset
from torch.utils.data import Subset, ConcatDataset, DataLoader
from training import train, valid_loop
from losses import VelocityLoss, PoseLoss
from utils import unflatten_pose
from pylie.torch import SO3

# TODO: estimate covariance of RMIs
# TODO: estimate scaling of RMIs

N = 50  # Window size
stride = 50
batch_size = 256
epochs = 1000
load_file = None
lr = 0.01
#torch.set_default_dtype(torch.float64)

# ##############################################################################
# Divide dataset into training and validation: take first 50 seconds as training.

raw_datasets = [
    RmiDataset("./data/processed/v1_03_difficult.csv", N, stride),
    RmiDataset("./data/processed/v2_01_easy.csv", N, stride),
    RmiDataset("./data/processed/v2_02_medium.csv", N, stride),
    RmiDataset("./data/processed/v2_03_difficult.csv", N, stride),
    RmiDataset("./data/processed/mh_02_easy.csv", N, stride),
    RmiDataset("./data/processed/mh_03_medium.csv", N, stride),
    RmiDataset("./data/processed/mh_04_difficult.csv", N, stride),
]

trainset_list = []
validset_list = []
for dataset in raw_datasets:
    idx = dataset.get_index_of_time(50)
    trainset_list.append(Subset(dataset, list(range(idx))))
    validset_list.append(Subset(dataset, list(range(idx, len(dataset)))))

trainset = ConcatDataset(trainset_list)
validset = ConcatDataset(validset_list)

""" Calibration"""
net = RmiModel()
loss_fn = PoseLoss()
load_file = "best_calibration.pt"
output_file = "calibration.pt"
lr = 0.0001

# TODO: there is something seriously wrong with calibration after the change
# to the preprocessing step

train(
    net,
    trainset=trainset,
    batch_size=batch_size,
    epochs=epochs,
    validset=validset,
    output_file=output_file,
    weights_file=load_file,
    lr=lr,
    loss_fn=loss_fn,
    use_gpu=False,
)
# print(" ANALYTICAL MODEL LOSS: " + str(model_loss))

print("done")