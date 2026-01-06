"""
Implementation of data setup in a script called data.py. 
The data was saved using torch.save, so to load it you should use torch.load.
"""


from pathlib import Path

import typer
from torch.utils.data import Dataset
from torchvision import transforms
import torch

raw_dir = "data/raw"


# Create a normalize transform
def normalize(images):
    return (images - images.mean()) / images.std()

def preprocess(raw_dir, processed_dir):
    """ Return train and test DaTALOADERS for corrupt MNIST"""

    train_images, train_target = [], []

    for i in range(6):
        test = ...
        train = ...

        #load train images
        image_file = Path(f"{raw_dir}/train_images_{i}.pt")
        target_file = Path(f"{raw_dir}/train_target_{i}.pt")
        
        if not image_file.exists() or not target_file.exists():
            print(f"Files not found. Contents of {raw_dir}:")
            for item in Path(raw_dir).iterdir():
                print(item)
            raise FileNotFoundError(f"Missing files for i={i}")
        
        train_images.append(torch.load(image_file))
        train_target.append(torch.load(target_file))
        
    # concatenate into torch object
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # load test data
    test_images = torch.load(f'{raw_dir}/test_images.pt')
    test_target = torch.load(f'{raw_dir}/test_target.pt')

    # unsqueeze images to add required ? first dimension and convert to float dtype
    train_images = train_images.unsqueeze(dim=1).float()
    test_images = test_images.unsqueeze(dim=1).float()
    
    # normalize images
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # convert target to 64-bit int dtype (required for certain loss functions)
    train_target = train_target.long()
    test_target = test_target.long()

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")

def corrupt_mnist(processed_dir):
    # load data 
    train_images = torch.load(f'{processed_dir}/train_images.pt')
    train_target = torch.load(f'{processed_dir}/train_target.pt')

    test_images = torch.load(f'{processed_dir}/test_images.pt')
    test_target = torch.load(f'{processed_dir}/test_target.pt')


    # convert to tensor dataset 
    train = torch.utils.data.TensorDataset(train_images, train_target)
    test = torch.utils.data.TensorDataset(test_images, test_target)

    return train, test


if __name__ == "__main__":
    preprocess(raw_dir="data/raw", processed_dir="data/processed")
    train_set, test_set = corrupt_mnist(processed_dir="data/processed")
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    
    # check that data is normalized
    print(f"Mean of training set: {train_set[0][0].mean().item()}")
    print(f"Std of training set: {train_set[0][0].std().item()}")
    print(f"Mean of test set: {test_set[0][0].mean().item()}")
    print(f"Std of test set: {test_set[0][0].std().item()}")

