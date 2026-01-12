"""
Docstring for tests.test_model. 

 Checks  that for a given input with shape X that the output of the model has shape Y.

When you implement a test you need to follow two standards
for pytest to be able to find your tests. 
First, any files created (except __init__.py) should always start with test_*.py.
Secondly, any test implemented needs to be wrapped into a function that again needs to start with test_*:
"""
import os 
import sys

from torch import nn
import torch

from mnist.model import Model

# # Add project root to path to allow imports when running directly
# _TEST_ROOT = os.path.dirname(__file__)
# _PROJECT_ROOT = os.path.dirname(_TEST_ROOT)
# _PATH_DATA = os.path.join(_PROJECT_ROOT, "data")

# this will be found and executed by pytest
def test_model():
    model = Model()
    x = torch.rand(1,1,28,28)

    y = model(x)
    assert y.shape == (1, 10)


if __name__ == "__main__":
    test_model()
    print("All tests passed!")

