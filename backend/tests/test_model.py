import pytest
import torch
from fastapi.testclient import TestClient
import json
import os
import sys
import logging
from pydantic import BaseModel
from typing import List, Dict, Union

from main import (
    app,
    create_dynamic_model,
    LayerConfig,
    NetworkConfig,
    DEFAULT_ARCHITECTURE_FILE,
    logger as main_logger
)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
sys.path.append(backend_dir)
logger.info(f"Backend directory: {backend_dir}")



client = TestClient(app)

def format_architecture(architecture):
    """Format architecture for readable logging"""
    return "\n".join([
        f"Layer {i+1}: {layer.type}"
        f"{' - ' + json.dumps(layer.params, indent=2) if layer.params else ''}"
        for i, layer in enumerate(architecture)
    ])

def load_default_architecture():
    """Load the default architecture from localStorage or fallback to best known architecture"""
    try:
        if not os.path.exists(DEFAULT_ARCHITECTURE_FILE):
            raise FileNotFoundError(
                f"Default architecture file not found at {DEFAULT_ARCHITECTURE_FILE}"
            )

        with open(DEFAULT_ARCHITECTURE_FILE, "r") as f:
            data = json.load(f)
            if "network_architecture" not in data:
                raise ValueError("Invalid architecture file format")
            architecture = [
                LayerConfig(**layer) for layer in data["network_architecture"]
            ]

            logger.info("\nTesting with architecture:")
            logger.info(format_architecture(architecture))

            return architecture
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        raise Exception(
            f"Error loading default architecture: {str(e)}\n"
            "Please set a default architecture from the frontend application "
            "first by clicking 'Set Default' on your best performing model."
        )

def test_model_creation():
    """Test if model can be created from default architecture"""
    logger.info("\n=== Testing Model Creation ===")
    architecture = load_default_architecture()
    model = create_dynamic_model(architecture)
    logger.info(f"Model created: {model}")
    batch_size = 64
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output = model(input_tensor)

    assert output.shape == (batch_size, 10), "Model output shape is incorrect"
    logger.info("Model creation successful")

def test_high_training_accuracy():
    """Test if model achieves >95% training accuracy"""
    logger.info("\n=== Testing Training Accuracy ===")
    architecture = load_default_architecture()

    training_config = NetworkConfig(
        network_architecture=architecture,
        optimizer="adam",
        learning_rate=0.01,
        epochs=1
    )

    response = client.post("/train", json=training_config.dict())
    assert response.status_code == 200

    results = response.json()
    final_train_accuracy = results["results"][-1]["train_accuracy"]

    logger.info(f"Training Accuracy: {final_train_accuracy * 100:.2f}%")
    assert final_train_accuracy > 0.95, f"Training accuracy ({final_train_accuracy * 100:.2f}%) should be > 95%"

def test_high_testing_accuracy():
    """Test if model achieves >95% testing accuracy"""
    logger.info("\n=== Testing Model Accuracy ===")
    architecture = load_default_architecture()
    logger.info(f"Architecture: {architecture}")
    training_config = NetworkConfig(
        network_architecture=architecture,
        optimizer="adam",
        learning_rate=0.01,
        epochs=1
    )

    response = client.post("/train", json=training_config.dict())
    assert response.status_code == 200

    results = response.json()
    final_test_accuracy = results["results"][-1]["test_accuracy"]

    logger.info(f"Test Accuracy: {final_test_accuracy * 100:.2f}%")
    assert final_test_accuracy > 0.95, f"Test accuracy ({final_test_accuracy * 100:.2f}%) should be > 95%"

def test_single_epoch_performance():
    """Test if model achieves required accuracy in single epoch"""
    logger.info("\n=== Testing Single Epoch Performance ===")
    architecture = load_default_architecture()

    training_config = NetworkConfig(
        network_architecture=architecture,
        optimizer="adam",
        learning_rate=0.01,
        epochs=1
    )

    response = client.post("/train", json=training_config.dict())
    assert response.status_code == 200

    results = response.json()
    assert len(results["results"]) == 1, "Should only train for one epoch"

    train_acc = results["results"][0]["train_accuracy"]
    test_acc = results["results"][0]["test_accuracy"]

    logger.info("Single Epoch Results:")
    logger.info(f"Train Accuracy: {train_acc * 100:.2f}%")
    logger.info(f"Test Accuracy: {test_acc * 100:.2f}%")
    assert train_acc > 0.95 and test_acc > 0.95, "Should achieve >95% accuracy in single epoch"

def test_model_params():
    """Test if model has less than 25000 parameters"""
    logger.info("\n=== Testing Model Parameters ===")
    architecture = load_default_architecture()
    model = create_dynamic_model(architecture)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Total Parameters: {total_params:,}")
    assert total_params < 25000, f"Model has {total_params:,} parameters, should be < 25,000"

def test_model_efficiency():
    """Comprehensive test for model efficiency"""
    logger.info("\n=== Testing Overall Model Efficiency ===")
    architecture = load_default_architecture()
    model = create_dynamic_model(architecture)
    total_params = sum(p.numel() for p in model.parameters())

    training_config = NetworkConfig(
        network_architecture=architecture,
        optimizer="adam",
        learning_rate=0.01,
        epochs=1
    )

    response = client.post("/train", json=training_config.dict())
    results = response.json()

    train_acc = results["results"][0]["train_accuracy"]
    test_acc = results["results"][0]["test_accuracy"]

    logger.info("\nModel Efficiency Summary:")
    logger.info("-------------------------")
    logger.info(f"Parameters: {total_params:,}")
    logger.info(f"Training Accuracy: {train_acc * 100:.2f}%")
    logger.info(f"Testing Accuracy: {test_acc * 100:.2f}%")
    logger.info(f"Epochs: 1")

    # Comprehensive assertions
    assert total_params < 25000, "Too many parameters"
    assert train_acc > 0.95, "Training accuracy too low"
    assert test_acc > 0.95, "Testing accuracy too low"
    assert len(results["results"]) == 1, "Too many epochs" 