import os 
import tempfile
import shutil

import torch
import pytest
from omegaconf import OmegaConf

from mnist.train import train
from mnist.model import Model

def test_train():
    """Test that training completes without errors with minimal epochs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train with minimal parameters for speed
        train(lr=1e-2, batch_size=64, epochs=1, models_path=tmpdir, seed=42)
        
        # Check that model was saved
        assert os.path.exists(f"{tmpdir}/model.pth"), "Model file was not saved"
        
        # Check that saved model can be loaded
        model = Model()
        model.load_state_dict(torch.load(f"{tmpdir}/model.pth"))
        assert model is not None, "Model could not be loaded"

def test_hydra_import():
    """ Test that config can be loaded correcty."""
    config = OmegaConf.load('config.yaml')

    # verify expected keys exist
    assert 'hyperparameters' in config, "Config missing hyperparameters section"
    assert 'lr' in config['hyperparameters'], "Config missing learning rate"
    assert 'batch_size' in config['hyperparameters'], "Config missing batch_size"
    assert 'epochs' in config['hyperparameters'], "Config missing epochs"
    assert 'seed' in config['hyperparameters'], "Config missing seed"
    
    # Verify types/ranges are reasonable
    assert config['hyperparameters']['lr'] > 0, "Learning rate should be positive"
    assert config['hyperparameters']['batch_size'] > 0, "Batch size should be positive"
    assert config['hyperparameters']['epochs'] > 0, "Epochs should be positive"



def test_optimizer_not_nan():
    """Test that training doesn't product NaN losses or weights"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train for a few steps 
        train(lr=1e-2, batch_size=64, epochs=1, models_path=tmpdir, seed=42)

        # load the trained model
        model = Model()
        state_dict = torch.load(f"{tmpdir}/model.pth")
        model.load_state_dict(state_dict)

        # Check that no parameters are NaN or Inf
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"Parameter {name} contains NaN values" 
            assert not torch.isinf(param).any(), f"Parameter {name} contains Inf values" 


def test_reproducibility_with_seed():
    """Test that same seed produces same results."""
    with tempfile.TemporaryDirectory() as tmpdir1, \
         tempfile.TemporaryDirectory() as tmpdir2:
        
        # Train twice with same seed
        seed = 999
        train(lr=1e-2, batch_size=64, epochs=1, models_path=tmpdir1, seed=seed)
        train(lr=1e-2, batch_size=64, epochs=1, models_path=tmpdir2, seed=seed)
        
        # Load both models
        model1 = Model()
        model1.load_state_dict(torch.load(f"{tmpdir1}/model.pth"))
        
        model2 = Model()
        model2.load_state_dict(torch.load(f"{tmpdir2}/model.pth"))
        
        # Check that parameters are identical
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            assert torch.allclose(param1, param2, rtol=1e-5), \
                f"Parameter {name1} differs between runs with same seed"


test_parameters = [
    (1e-3,32,1,42),
    (1e-2,64,2,24),
    (1e-2,128,1,None)
]

@pytest.mark.parametrize("lr,batch_size,epochs,seed", test_parameters)
def test_different_hyperparameters(lr, batch_size, epochs, seed):
    """Test that training works with different hyperparameter values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with different learning rates
        train(lr=lr, batch_size=batch_size, epochs=epochs, seed=seed, models_path=tmpdir)
        assert os.path.exists(f"{tmpdir}/model.pth")