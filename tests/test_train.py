import os
from src.train import train_model


def test_train_model():
    # Train model
    train_model()

    # Check if model file is created
    assert os.path.exists("model.pkl"), "Model file not created!"
