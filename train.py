from math import floor
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch
from model import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)


def train(
    net,
    trainset,
    output_filename="training_results",
    epochs=10,
    batch_size=1000,
    validation_set=None,
    compare_model=None,
    weights_file=None,
    output_file="rminet_weights.pth",
):

    if weights_file is not None:
        filename = "./results/" + weights_file
        net.load_state_dict(torch.load(filename))

    writer = SummaryWriter(flush_secs=1)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=0, drop_last=True
    )

    if validation_set is not None:
        validloader = DataLoader(
            validation_set, batch_size=batch_size, num_workers=0, drop_last=True
        )

    # criterion = torch.nn.MSELoss()
    criterion = pose_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-5)

    # Training
    model_loss = 0.0
    for epoch in range(epochs):

        # Stochastic Mini-batches
        running_loss = 0.0
        if criterion is pose_loss:
            running_C_loss = 0.0
            running_v_loss = 0.0
            running_r_loss = 0.0

        for i, training_sample in enumerate(trainloader, 0):
            # Get minibatch raw data
            x_train, y_train = training_sample

            # Use neural network to predict RMIs
            y_predict = net(x_train[:, 1:, :])
            # Evaluate (custom) loss function

            if criterion is pose_loss:
                loss, info = criterion(y_predict, y_train, with_info=True)
            else:
                loss = criterion(y_predict, y_train)

            # Perform an SGD step
            optimizer.zero_grad()
            if epoch != 0:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            if criterion is pose_loss:
                running_C_loss += info["C_loss"]
                running_v_loss += info["v_loss"]
                running_r_loss += info["r_loss"]

            # If an alternate model was fed as a comparison point, evaluate the
            # loss once.
            if compare_model is not None and epoch == 0:
                y_compare = compare_model(x_train)
                loss = criterion(y_compare, y_train)
                model_loss += loss.item()
                print(loss)

        running_loss /= i + 1
        if criterion is pose_loss:
            running_C_loss /= i + 1
            running_v_loss /= i + 1
            running_r_loss /= i + 1

        if compare_model is not None and epoch == 0:
            model_loss /= i + 1

        # Calculate validation loss
        if validation_set is not None:
            running_vloss = 0.0
            if criterion is pose_loss:
                running_C_vloss = 0.0
                running_v_vloss = 0.0
                running_r_vloss = 0.0
            for i, validation_sample in enumerate(validloader):
                vx, vy = validation_sample
                vpredict = net(vx[:, 1:, :])
                if criterion is pose_loss:
                    vloss, info = criterion(vpredict, vy, with_info=True)
                else:
                    vloss = criterion(vpredict, vy)

                running_vloss += vloss

                if criterion is pose_loss:
                    running_C_vloss += info["C_loss"]
                    running_v_vloss += info["v_loss"]
                    running_r_vloss += info["r_loss"]

            running_vloss /= i + 1
            if criterion is pose_loss:
                running_C_vloss /= i + 1
                running_v_vloss /= i + 1
                running_r_vloss /= i + 1
        else:
            running_vloss = 0

        print(
            "Epoch: %d, Running Loss: %.3f, Validation Loss: %.3f, Analytical Model Loss: %.3f"
            % (epoch, running_loss, running_vloss, model_loss)
        )

        writer.add_scalars(
            "Loss/Total",
            {"Training": running_loss, "Validation": running_vloss},
            epoch,
        )

        if criterion is pose_loss:
            writer.add_scalars(
                "Loss/Rotation",
                {"Training": running_C_loss, "Validation": running_C_vloss},
                epoch,
            )

            writer.add_scalars(
                "Loss/Velocity",
                {"Training": running_v_loss, "Validation": running_v_vloss},
                epoch,
            )

            writer.add_scalars(
                "Loss/Position",
                {"Training": running_r_loss, "Validation": running_r_vloss},
                epoch,
            )

        writer.flush()
        torch.save(net.state_dict(), "./results/" + output_file)


if __name__ == "__main__":
    N = 200  # Window size
    stride = 10
    trainset1 = RmiDataset("./data/processed/v1_01_easy.csv", N, stride)
    trainset2 = RmiDataset("./data/processed/v1_03_difficult.csv", N, stride)
    trainset3 = RmiDataset("./data/processed/v2_01_easy.csv", N, stride)
    trainset4 = RmiDataset("./data/processed/v2_02_medium.csv", N, stride)
    trainset5 = RmiDataset("./data/processed/v2_03_difficult.csv", N, stride)
    trainset = ConcatDataset([trainset1, trainset2, trainset3, trainset4, trainset5])

    validset = RmiDataset("./data/processed/v1_02_medium.csv", N, stride)

    net = RmiNet(window_size=N)
    model = RmiModel()

    train(
        net,
        trainset,
        batch_size=100,
        epochs=1000,
        validation_set=validset,
        compare_model=None,
        output_file="rminet_weights.pt",
    )

    print("done")
