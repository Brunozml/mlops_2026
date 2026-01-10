"""
@ Saturday Jan 10th wrote from scratch. yay

TODO: make into a class
"""

import os 

from torch.utils.data import Dataset
from torchvision import transforms
import torch


raw_dir = "data/raw"
processed_dir = "data/processed"

def normalize(images):
    """ Normalize images to have zero mean and unit variance """
    return (images - images.mean()) / images.std()

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    
    # load train data
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(
            torch.load(f"{raw_dir}/train_images_{i}.pt")
        )
        train_target.append(
            torch.load(f"{raw_dir}/train_target_{i}.pt")
        )
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # load test data
    test_images = torch.load(f"{raw_dir}/test_images.pt")
    test_target = torch.load(f"{raw_dir}/test_target.pt")

    # unsqueeze from [N_samples , width, height] to [N_samples, 1, width, height]
    train_images = train_images.unsqueeze(dim=1) 
    test_target = test_images.unsqueeze(dim=1)

    # convert target to 64-bit int dtype (required for certain loss functions)
    train_target = train_target.long()
    test_target = test_target.long()

    # normalize to [0, 1]
    train_images = train_images.float() / 255.0
    test_images = test_images.float() / 255.0

    # save to processed directory
    os.makedirs(processed_dir, exist_ok=True)
    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")

    
    return train_images, train_target, test_images, test_target

if __name__ == "__main__":
    train_images, train_target, test_images, test_target = corrupt_mnist()
    print(f"Train shape: {train_images.shape}, Test shape: {test_images.shape}")
