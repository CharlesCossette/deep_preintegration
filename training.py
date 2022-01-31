from utils import count_parameters
from copy import deepcopy
from torch.utils.data import DataLoader
import torch
from dataset import add_noise
from torch.utils.tensorboard import SummaryWriter


def train_loop(net, trainloader, optimizer, criterion, device="cpu"):
    # Stochastic Mini-batches
    running_loss = 0.0
    net.train()
    for i, training_sample in enumerate(trainloader, 0):
        # Get minibatch raw data
        x_train, y_train = training_sample
        x_train = add_noise(x_train) # TODO: move to dataset
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        # Use neural network to predict RMIs
        y_predict = net(x_train)
        loss = criterion(y_predict, y_train)

        # Perform an SGD step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    running_loss /= i + 1
    return running_loss


def valid_loop(net, validloader, criterion, device="cpu"):
    # Calculate validation loss
    running_vloss = 0.0
    if hasattr(net, "eval"):
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
    loss_fn=torch.nn.MSELoss(),
    use_gpu=False,
    lr=0.01,
    weight_decay=1e-5,
):

    if weights_file is not None:
        filename = "./results/" + weights_file
        net.load_state_dict(torch.load(filename, map_location="cpu"))

    if torch.cuda.is_available() and use_gpu:
        device = "cuda"
        net.to(device)
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

    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-4
    )
    #scheduler = None
    # TODO: make scheduler an argument
    # TODO: ? make optimizer an argument

    # Training
    valid_loss = 0.0
    best_valid_loss = 1e10
    for epoch in range(epochs):

        training_loss = train_loop(net, trainloader, optimizer, criterion, device)

        if scheduler is not None:
            scheduler.step(epoch)
            writer.add_scalar("Learning Rate", torch.Tensor(scheduler.get_last_lr()), epoch)

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

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(net.state_dict(), "./results/best_" + output_file)
            print("New lowest validation loss!")

    writer.close()



