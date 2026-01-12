import os

from torch.utils.data import Dataset
import torch

from mnist.data import corrupt_mnist

def test_data():
    train, test = corrupt_mnist()

    #  assert len(dataset) == N_train for training and N_test for test
    assert len(train) == 30_000
    assert len(test) == 5_000
    # checks on each individual datapoint
    for dataset in [train, test]:
        for x, y in dataset:
            #assert that each datapoint has shape [1,28,28]
            assert x.shape == (1, 28, 28)

             # assert that all labels are in range
            assert y in range(10)
    
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()
    

    



if __name__ == "__main__":
    test_data()
    print("All tests passed!")