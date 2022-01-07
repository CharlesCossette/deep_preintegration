from math import floor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import numpy as np
import pandas as pd
from pylie import SO3

# TODO: check gravity direction


class RmiDataset(Dataset):
    def __init__(self, filename, window_size=200):
        self._file = open(filename, "r")
        self._df = pd.read_csv(filename, sep=",", header=None)
        self._window_size = window_size
        pass

    def __len__(self):
        return floor(self._df.shape[0] / self._window_size)

    def __getitem__(self, idx):
        range_start = idx * self._window_size
        range_stop = (idx + 1) * self._window_size

        if range_stop > self._df.shape[0]:
            raise RuntimeError("programming error")

        sample_data = self._df[range_start:range_stop].to_numpy()
        t_data = sample_data[:, 0]
        gyro_data = sample_data[:, 1:4]
        accel_data = sample_data[:, 4:7]

        DT = t_data[-1] - t_data[0]
        g_a = np.array([0, 0, -9.80665]).reshape((-1, 1))
        r_zw_a_i = sample_data[0, 7:10].reshape((-1, 1))
        r_zw_a_j = sample_data[-1, 7:10].reshape((-1, 1))
        v_zw_a_i = sample_data[0, 10:13].reshape((-1, 1))
        v_zw_a_j = sample_data[-1, 10:13].reshape((-1, 1))
        C_ab_i = sample_data[0, 13:]
        C_ab_j = sample_data[-1, 13:]
        C_ab_i = C_ab_i.reshape((3, 3))
        C_ab_j = C_ab_j.reshape((3, 3))

        DC = C_ab_i.T @ C_ab_j
        DV = C_ab_i.T @ (v_zw_a_j - v_zw_a_i - DT * g_a)
        DR = C_ab_i.T @ (
            r_zw_a_j - r_zw_a_i - v_zw_a_i * DT - 0.5 * g_a * DT ** 2
        )

        x = torch.from_numpy(np.vstack((t_data.T, gyro_data.T, accel_data.T)))
        x = x.to(torch.float32)

        y = torch.from_numpy(
            np.hstack((DC.flatten(), DV.flatten(), DR.flatten()))
        )
        y = y.to(torch.float32)
        return x, y


class RmiNet(torch.nn.Module):
    def __init__(self, window_size=200):
        super(RmiNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(6, 1, 5, dtype=torch.float32)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.LazyLinear(50)
        self.leaky1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(50, 30)
        self.leaky2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(30, 9 + 3 + 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.leaky1(self.linear1(x))
        x = self.leaky2(self.linear2(x))
        x = self.linear3(x)

        # Normalize the rotation matrix to make it a valid element of SO(3)
        R = torch.reshape(x[:, 0:9], (x.shape[0], 3, 3))
        U, _, VT = torch.linalg.svd(R)
        S = torch.eye(3).reshape((1, 3, 3)).repeat(x.shape[0], 1, 1)
        S[:, 2, 2] = torch.det(torch.matmul(U, VT))
        R_norm = torch.matmul(U, torch.matmul(S, VT))
        R_flat = torch.reshape(R_norm, (x.shape[0], 9))

        # TODO: double check to see if the below operation works as intended
        return torch.cat((R_flat, x[:, 9:]), 1)


class RmiModel(torch.nn.Module):
    def __init__(self, window_size=200):
        super(RmiModel, self).__init__()

    def forward(self, x):
        # shape[0] is the batch size
        y = torch.zeros((x.shape[0], 15))
        for idx in range(x.shape[0]):
            y[idx, :] = self._get_rmis(x[idx, :, :])

        return y

    def _get_rmis(self, x):
        t = x[0, :].detach().to("cpu").numpy()
        gyro = x[1:4, :].detach().to("cpu").numpy()
        accel = x[4:7, :].detach().to("cpu").numpy()

        DC = np.identity(3)
        DV = np.zeros((3, 1))
        DR = np.zeros((3, 1))

        for idx in range(1, x.shape[1]):
            dt = t[idx] - t[idx - 1]
            w = gyro[:, idx - 1].reshape((-1, 1))
            a = accel[:, idx - 1].reshape((-1, 1))
            DR += DV * dt + 0.5 * DC @ a * dt ** 2
            DV += DC @ a * dt
            DC = DC @ SO3.Exp(w * dt)

        temp = np.hstack((DC.flatten(), DV.flatten(), DR.flatten()))
        return torch.from_numpy(temp)


def train(
    net,
    trainset,
    output_filename="training_results",
    epochs=10,
    batch_size=1000,
    validation_set=None,
    compare_model=None,
):

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    if validation_set is not None:
        validloader = DataLoader(
            validation_set, batch_size=batch_size, shuffle=False, num_workers=0
        )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    # Training
    model_loss = 0.0
    for epoch in range(epochs):

        # Stochastic Mini-batches
        running_loss = 0.0

        for i, training_sample in enumerate(trainloader, 0):
            x_train, y_train = training_sample
            x_train = torch.autograd.Variable(x_train)
            y_train = torch.autograd.Variable(y_train)

            optimizer.zero_grad()

            y_predict = net(x_train[:, 1:, :])
            loss = criterion(y_predict, y_train)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if compare_model is not None and epoch == 0:
                y_compare = compare_model(x_train)
                loss = criterion(y_compare, y_train)
                model_loss += loss.item()
                print(loss)

        # Calculate validation loss
        if validation_set is not None:
            running_vloss = 0.0
            for i, validation_sample in enumerate(validloader):
                vx, vy = validation_sample
                vpredict = net(vx[:, 1:, :])
                vloss = criterion(vpredict, vy)
                running_vloss += vloss
        else:
            running_vloss = 0

        print(
            "Epoch: %d, Running Loss: %.3f, Validation Loss: %.3f, Analytical Model Loss: %.3f"
            % (epoch, running_loss, running_vloss, model_loss)
        )


if __name__ == "__main__":
    N = 200  # Window size
    trainset1 = RmiDataset("./data/processed/v1_01_easy.csv", N)
    trainset2 = RmiDataset("./data/processed/v1_02_medium.csv", N)
    trainset = ConcatDataset([trainset1, trainset2])
    net = RmiNet(window_size=N)
    model = RmiModel()

    train(
        net,
        trainset,
        batch_size=5,
        epochs=200,
        validation_set=None,
        compare_model=model,
    )
    print("done")
