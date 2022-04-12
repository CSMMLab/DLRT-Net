import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary

from dlraNet import DLRANet, PartialDLRANet


def main():
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # build the network
    batch_size = 64
    input_dim = 784
    output_dim = 10
    layer_width = 100
    layer_num = 0
    low_rank = 50

    num_layers_normal = 4
    layer_width_normal = 100

    model = PartialDLRANet(input_dim=input_dim, output_dim=output_dim, layer_width_dlra=layer_width,
                           num_layers_dlra=layer_num, low_rank=low_rank, layer_width_normal=layer_width_normal,
                           num_layers_normal=num_layers_normal)
    # model.svd_initialization()
    # model = model.to(device)
    # --- summary of the model
    # summary(model, input_size=(28, 28))

    # print params
    # for param in model.parameters():
    #    print(param.data)

    loss_fn = nn.CrossEntropyLoss()

    # train the network
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")

    # evaluate
    # save_and_eval(model, test_data)
    # check weights for singular values
    # svd_inspection(model)

    return 0


def train(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    # model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        x = nn.Flatten()(X)

        # Compute prediction error

        ## K-Step ##
        # print("K-Step")
        out = model.k_step_forward(x)
        loss = loss_fn(out, y)
        # print(loss)
        model.optim_K.zero_grad()
        model.optim_b.zero_grad()
        model.optim_Wb.zero_grad()
        model.optim_W.zero_grad()
        loss.backward()
        model.k_step_update()
        model.w_and_b_update()
        # model.print_weights_K()
        # model.clear_grads()

        ## L-Step ##
        # print("L-Step")
        out = model.l_step_forward(x)
        loss = loss_fn(out, y)
        # print(loss)
        model.optim_Lt.zero_grad()
        loss.backward()
        model.l_step_update()
        # model.print_weights_Lt()
        # model.clear_grads()

        ## S-Step ##
        # print("S-Step")
        out = model.s_step_forward(x)
        # model.print_weights_S()

        loss = loss_fn(out, y)
        # print(loss)
        model.optim_S.zero_grad()
        loss.backward()
        # model.print_weights_S()
        # model.print_aux_M()
        # model.print_aux_N()
        model.s_step_update()
        # model.clear_grads()

        # print(model.weights[1])
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            x = nn.Flatten()(X)
            ## Only S-Step ##
            pred = model.s_step_forward(x)
            ## L-Step ##
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def save_and_eval(model, test_data):
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
               "Ankle boot", ]
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
    return 0


def svd_inspection(model):
    # perform svd for all weight matrices
    list_svdParams = []
    for param in model.parameters():
        if len(param.size()) > 1:  # skip bias terms
            U, S, Vh = torch.svd(param, some=True)
            list_svdParams.append([U, S, Vh])

    # check the S matrices
    for decomp in list_svdParams:
        print(decomp[1].size())

    # print the matrices to file
    count = 0
    for decomp in list_svdParams:
        torch.save(decomp[0], 'mat/U_' + str(count) + '.pt')
        torch.save(decomp[1], 'mat/S_' + str(count) + '.pt')
        torch.save(decomp[2], 'mat/V_' + str(count) + '.pt')
        count += 1

    return 0


def inspect_matrices():
    for i in range(0, 3):
        U = torch.load('mat/U_' + str(i) + '.pt')
        S = torch.load('mat/S_' + str(i) + '.pt')
        V = torch.load('mat/V_' + str(i) + '.pt')

        print("layer: " + str(i) + " maxSV: " +
              str(torch.max(S)) + " minSV: " + str(torch.min(S)))
        print(U.size())
        print(S.size())
        print(V.size())
        print(S)

        print("---------")

    return 0


def load_model_save_matrices():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # model = NeuralNetwork().to(device)
    # model.load_state_dict(torch.load("model.pth"))

    # print the matrices to file
    # svd_inspection(model)
    return 0


if __name__ == '__main__':
    main()
    # load_model_save_matrices()
    inspect_matrices()
