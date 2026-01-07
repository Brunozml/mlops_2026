import os

import matplotlib.pyplot as plt
import torch
import typer
from cnn_mnist.model import Model
from cnn_mnist.data import corrupt_mnist

# specify training device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr = 1e-3, batch_size = 32, epochs = 1, models_path = "models/"):
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # ensure output directories exist
    os.makedirs(models_path, exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    dataset, _ = corrupt_mnist("data/processed")
    model = Model().to(DEVICE)
    
    # instantiate data loader
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # define criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # record training statistics for later use
    statistics = {
        "train_loss": [],
        "train_accuracy": []
    }

    # epoch loop
    for epoch in range(epochs):

        # batch loop
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            # set gradients to zero
            optimizer.zero_grad()
            # forward pass
            out =  model(img)

            # calculate loss
            loss = criterion(out, target)

            # backward pass
            loss.backward()

            # optimization step
            optimizer.step()

            # calculate training statistics
            statistics["train_loss"].append(loss.item())
            accuracy = (out.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
    
    # save model
    torch.save(model.state_dict(), f"{models_path}/model.pth")
    print(f"Training complete! model saved to {models_path}/model.pth")

    # create and save training visual
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    
if __name__ == "__main__":
    train()
