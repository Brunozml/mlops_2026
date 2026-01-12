from torch import nn
import torch

class Model(nn.Module):
    """
    Convolutional Neural Network with 3 convolutional layers, 
    one fully connected layer, max_pooling, and relu activation functions.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=1
                    )
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1
                    )
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1
                    )
        
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)
    

if __name__ == "__main__":
    model = Model()
    print(f"Model Architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    x = torch.rand(1,1,28,28)
    print(f"Output shape of model: {model(x).shape}")