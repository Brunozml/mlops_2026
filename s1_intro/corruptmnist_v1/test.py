"""
Script objective: Identify corruption applied to MNIST dataset qualitatively
"""
import helper
import torch
from torchvision import transforms

# Define a transform function to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the training data
train_images = []
train_targets = []

# Load all chunks of training data
for i in range(6):
    train_images.append(torch.load(f'train_images_{i}.pt'))
    train_targets.append(torch.load(f'train_target_{i}.pt'))

# Concatenate all chunks
train_images = torch.cat(train_images, dim=0)
train_targets = torch.cat(train_targets, dim=0)

print(f"Training data shape: {train_images.shape}")
print(f"Training targets shape: {train_targets.shape}")

# Create a custom Dataset class
from torch.utils.data import Dataset, DataLoader

class CustomMNISTDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

# Create dataset and dataloader
train_dataset = CustomMNISTDataset(train_images, train_targets)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(f"Number of batches: {len(train_loader)}")

# Visualize some examples
import matplotlib.pyplot as plt

# Get one batch
batch_images, batch_labels = next(iter(train_loader))
print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")

fig, axes = plt.subplots(4, 5, figsize=(15, 18))
axes = axes.ravel()

for i in range(20):
    # MNIST images are grayscale, so squeeze to 2D for display
    axes[i].imshow(batch_images[i].squeeze(), cmap='gray')
    axes[i].set_title(f"Label: {batch_labels[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

