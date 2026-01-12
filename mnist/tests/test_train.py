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
