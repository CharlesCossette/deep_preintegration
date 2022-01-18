from model import RmiDataset, DeltaTransRmiNet, DeltaRotRmiNet
from utils import count_parameters
from losses import DeltaRotRmiLoss, DeltaTransRmiLoss
from copy import deepcopy
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch
from torch.utils.tensorboard import SummaryWriter


def train_loop(net, trainloader, optimizer, criterion, device="cpu"):
    # Stochastic Mini-batches
    running_loss = 0.0
    net.train()
    for i, training_sample in enumerate(trainloader, 0):
        # Get minibatch raw data
        x_train, y_train = training_sample
        x_train = x_train.to(device)
        y_train = y_train.to(device)

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


def valid_loop(net, validloader, criterion, device="cpu"):
    # Calculate validation loss
    running_vloss = 0.0
    net.eval()
    with torch.no_grad():
        for i, validation_sample in enumerate(validloader, 0):
            vx, vy = validation_sample
            vx = vx.to(device)
            vy = vy.to(device)

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
    use_gpu=False,
):

    if weights_file is not None:
        filename = "./results/" + weights_file
        net.load_state_dict(torch.load(filename))

    if torch.cuda.is_available() and use_gpu:
        device = "cuda"
        net.cuda()
        print("Using GPU for training.")
    else:
        device = "cpu"
        print("Using CPU for training.")

    writer = SummaryWriter(flush_secs=1)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    if validset is not None:
        validloader = DataLoader(validset, batch_size=batch_size)

    criterion = loss_fn
    criterion_valid = deepcopy(loss_fn)

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.05, weight_decay=1e-5)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-2)

    # Training
    # training_loss = valid_loop(net, trainloader, criterion, device)

    valid_loss = 0.0
    for epoch in range(epochs):

        training_loss = train_loop(net, trainloader, optimizer, criterion, device)

        if hasattr(criterion, "write_info"):
            criterion.write_info(writer, epoch, "Training")

        if validset is not None:
            valid_loss = valid_loop(net, validloader, criterion_valid, device)
            if hasattr(criterion_valid, "write_info"):
                criterion_valid.write_info(writer, epoch, "Validation")

        if epoch == 0:
            print(
                "Training Neural Network with "
                + str(count_parameters(net))
                + " parameters."
            )

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
    N = 500  # Window size
    stride = 20
    batch_size = 128
    epochs = 1000
    load_file = None

    # torch.set_default_dtype(torch.float64)
    with_model = True
    raw_datasets = [
        RmiDataset("./data/processed/v1_03_difficult.csv", N, stride, with_model),
        RmiDataset("./data/processed/v2_01_easy.csv", N, stride, with_model),
        RmiDataset("./data/processed/v2_02_medium.csv", N, stride, with_model),
        RmiDataset("./data/processed/v2_03_difficult.csv", N, stride, with_model),
        RmiDataset("./data/processed/v1_02_medium.csv", N, stride, with_model),
    ]

    trainset_list = []
    validset_list = []
    for dataset in raw_datasets:
        idx = dataset.get_index_of_time(50)
        trainset_list.append(Subset(dataset, list(range(idx))))
        validset_list.append(Subset(dataset, list(range(idx, len(dataset)))))

    trainset = ConcatDataset(trainset_list)
    validset = ConcatDataset(validset_list)

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
    net = DeltaTransRmiNet(window_size=N)
    loss_fn = DeltaTransRmiLoss()
    # load_file = "dtransrminet_weights.pt"
    output_file = "dtransrminet_weights.pt"

    # """ Rot only"""
    # net = DeltaRotRmiNet(window_size=N)
    # loss_fn = DeltaRotRmiLoss()
    # output_file = "drotrminet_weights.pt"
    # # load_file = output_file

    train(
        net,
        trainset=trainset,
        batch_size=batch_size,
        epochs=epochs,
        validset=validset,
        output_file=output_file,
        weights_file=load_file,
        loss_fn=loss_fn,
        use_gpu=True,
    )
    # print(" ANALYTICAL MODEL LOSS: " + str(model_loss))

    print("done")
