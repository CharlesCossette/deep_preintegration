from model import *
from utils import *
from losses import *
from copy import deepcopy
from math import floor
from h11 import Data
from torch.utils.data import DataLoader, ConcatDataset
import torch
from torch.utils.tensorboard import SummaryWriter


def train_loop(net, trainloader, optimizer, criterion):
    # Stochastic Mini-batches
    running_loss = 0.0
    for i, training_sample in enumerate(trainloader, 0):
        # Get minibatch raw data
        x_train, y_train = training_sample

        # Use neural network to predict RMIs
        y_predict = net(x_train)
        loss = criterion(y_predict, y_train)

        # Perform an SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    running_loss /= i + 1
    return running_loss


def valid_loop(net, validloader, criterion):
    # Calculate validation loss
    running_vloss = 0.0
    for i, validation_sample in enumerate(validloader, 0):
        vx, vy = validation_sample
        vpredict = net(vx)
        vloss = criterion(vpredict, vy)

        running_vloss += vloss

    running_vloss /= i + 1
    return running_vloss


def train(
    net,
    trainset,
    epochs=10,
    batch_size=1000,
    validset=None,
    weights_file=None,
    output_file="rminet_weights.pth",
    loss_fn=torch.nn.MSELoss,
):
    if weights_file is not None:
        filename = "./results/" + weights_file
        net.load_state_dict(torch.load(filename))

    print("Using CPU for training.")

    writer = SummaryWriter(flush_secs=1)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=0, shuffle=True
    )

    if validset is not None:
        validloader = DataLoader(validset, batch_size=batch_size, num_workers=0)

    criterion = loss_fn
    criterion_valid = deepcopy(loss_fn)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0)
    # optimizer = torch.optim.Adam(net.parameters())

    # Training
    training_loss = valid_loop(net, trainloader, criterion)
    print("Training Neural Network with " + str(count_parameters(net)) + " parameters.")
    valid_loss = 0.0
    for epoch in range(epochs):

        training_loss = train_loop(net, trainloader, optimizer, criterion)

        if hasattr(criterion, "write_info"):
            criterion.write_info(writer, epoch, "Training")

        if validset is not None:
            valid_loss = valid_loop(net, validloader, criterion_valid)
            if hasattr(criterion_valid, "write_info"):
                criterion_valid.write_info(writer, epoch, "Validation")

        print(
            "Epoch: %d, Running Loss: %.6f, Validation Loss: %.6f"
            % (epoch, training_loss, valid_loss)
        )

        writer.add_scalars(
            "Loss/Total",
            {"Training": training_loss, "Validation": valid_loss},
            epoch,
        )

        writer.flush()

        torch.save(net.state_dict(), "./results/" + output_file)
    writer.close()


if __name__ == "__main__":
    N = 1000  # Window size
    stride = 999
    batch_size = 1
    epochs = 1000
    load_file = None

    # net = RmiNet(window_size=N)
    # trainset1 = RmiDataset("./data/processed/v1_01_easy.csv", N, stride)
    # trainset2 = RmiDataset("./data/processed/v1_03_difficult.csv", N, stride)
    # trainset3 = RmiDataset("./data/processed/v2_01_easy.csv", N, stride)
    # trainset4 = RmiDataset("./data/processed/v2_02_medium.csv", N, stride)
    # trainset5 = RmiDataset("./data/processed/v2_03_difficult.csv", N, stride)
    # trainset = ConcatDataset([trainset1, trainset2, trainset3, trainset4, trainset5])
    # validset = RmiDataset("./data/processed/v1_02_medium.csv", N, stride)

    #
    # train(
    #     net,
    #     trainset=trainset1,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     validset=validset,
    #     compare_model=model,
    #     output_file="rminet_weights.pt",
    #     weights_file="rminet_weights.pt",
    #     loss_fn=pose_loss
    # )

    torch.set_default_dtype(torch.float64)
    trainset1 = RmiDataset("./data/processed/v1_01_easy.csv", N, stride, True)
    trainset2 = RmiDataset("./data/processed/v1_03_difficult.csv", N, stride, True)
    trainset3 = RmiDataset("./data/processed/v2_01_easy.csv", N, stride, True)
    trainset4 = RmiDataset("./data/processed/v2_02_medium.csv", N, stride, True)
    trainset5 = RmiDataset("./data/processed/v2_03_difficult.csv", N, stride, True)
    trainset = ConcatDataset([trainset1, trainset2, trainset3, trainset4, trainset5])
    validset = RmiDataset("./data/processed/v1_02_medium.csv", N, stride, True)

    # Evaluate Analytical Model
    # modelset = RmiDataset("./data/processed/v1_01_easy.csv", N, stride, False)
    # modelloader = DataLoader(modelset, batch_size=batch_size)
    # model = RmiModel()
    # model_loss = valid_loop(model, modelloader, pose_loss)
    # print(" ANALYTICAL MODEL LOSS: " + str(model_loss))

    """ Joint trans/rot. """
    # net = DeltaRmiNet(window_size=N)
    # loss_fn = delta_rmi_loss

    """ Trans only"""
    # net = DeltaTransRmiNet(window_size=N)
    # loss_fn = DeltaTransRmiLoss()
    # load_file = "dtransrminet_weights.pt"
    # output_file = "dtransrminet_weights.pt"

    """ Rot only"""
    net = DeltaRotRmiNet(window_size=N)
    loss_fn = DeltaRotRmiLoss()
    output_file = "drotrminet_weights.pt"
    # load_file = output_file

    train(
        net,
        trainset=trainset1,
        batch_size=batch_size,
        epochs=epochs,
        validset=validset,
        output_file=output_file,
        weights_file=load_file,
        loss_fn=loss_fn,
    )
    # print(" ANALYTICAL MODEL LOSS: " + str(model_loss))

    print("done")
