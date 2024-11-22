from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import numpy as np
import logging
import traceback
import json
from datetime import datetime
import os
from models.mnist_model import MNISTNet
from PIL import Image
import io
import base64
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LayerConfig(BaseModel):
    type: str
    params: Dict[str, Union[int, float, str]]

class AugmentationConfig(BaseModel):
    enabled: bool
    rotation: float
    zoom: float
    width_shift: float
    height_shift: float
    horizontal_flip: bool

class NetworkConfig(BaseModel):
    network_architecture: List[LayerConfig]
    optimizer: str
    learning_rate: float
    epochs: int
    augmentation: AugmentationConfig

class TrainingResult(BaseModel):
    epoch: int
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float

# Get the absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Backend directory: {BACKEND_DIR}")

# Set paths relative to backend directory
DEFAULT_ARCHITECTURE_FILE = os.path.join(BACKEND_DIR, "tests", "default_architecture.json")
logger.info(f"Default architecture file path: {DEFAULT_ARCHITECTURE_FILE}")

HISTORY_FILE = os.path.join(BACKEND_DIR, "training_history.json")
logger.info(f"History file path: {HISTORY_FILE}")

def load_training_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
                logger.info(f"Successfully loaded history with {len(history)} entries")
                return history
        logger.info("No history file found, returning empty list")
        return []
    except Exception as e:
        logger.error(f"Error loading training history: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def save_training_history(history):
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Successfully saved history with {len(history)} entries")
    except Exception as e:
        logger.error(f"Error saving training history: {str(e)}")
        logger.error(traceback.format_exc())

def create_model(conv_layers):
    try:
        logger.info(f"Creating model with layers: {conv_layers}")
        if not conv_layers:
            raise ValueError("No convolutional layers provided")
        model = MNISTNet(conv_layers)
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def apply_augmentation(image, config):
    """Apply augmentation to a single image"""
    try:
        img_pil = TF.to_pil_image(image)
        
        if config.rotation > 0:
            angle = random.uniform(-config.rotation, config.rotation)
            img_pil = TF.rotate(img_pil, angle)
        
        if config.zoom > 0:
            scale = 1.0 + random.uniform(0, config.zoom)
            new_size = [int(dim * scale) for dim in img_pil.size]
            img_pil = TF.resize(img_pil, new_size)
            # Crop back to original size
            i = (img_pil.size[0] - 28) // 2
            j = (img_pil.size[1] - 28) // 2
            img_pil = TF.crop(img_pil, j, i, 28, 28)
        
        if config.width_shift > 0 or config.height_shift > 0:
            width_shift = int(28 * random.uniform(-config.width_shift, config.width_shift))
            height_shift = int(28 * random.uniform(-config.height_shift, config.height_shift))
            img_pil = TF.affine(img_pil, 0, [width_shift, height_shift], 1, 0)
        
        if config.horizontal_flip and random.random() > 0.5:
            img_pil = TF.hflip(img_pil)
        
        # Convert back to tensor
        return TF.to_tensor(img_pil)
    except Exception as e:
        logger.error(f"Error in augmentation: {str(e)}")
        return image

def train_epoch(model, train_loader, optimizer, criterion, augmentation_config=None):
    try:
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                # Apply augmentation if enabled
                if augmentation_config and augmentation_config.enabled:
                    augmented_data = torch.stack([
                        apply_augmentation(img, augmentation_config) 
                        for img in data
                    ])
                    data = augmented_data

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if batch_idx % 100 == 0:
                    logger.info(f'Training batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
        return correct / total, total_loss / len(train_loader)
    except Exception as e:
        logger.error(f"Error in train_epoch: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def test_epoch(model, test_loader, criterion):
    try:
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return correct / total, test_loss / len(test_loader)
    except Exception as e:
        logger.error(f"Error in test_epoch: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def load_data():
    try:
        logger.info("Loading MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)
        
        logger.info("Dataset loaded successfully")
        return train_loader, test_loader
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.post("/train")
async def train_model(config: NetworkConfig):
    try:
        start_time = datetime.now()
        logger.info(f"Starting training with config: {config}")
        
        # Log augmentation settings if enabled
        if config.augmentation and config.augmentation.enabled:
            logger.info("Data augmentation enabled with settings:")
            logger.info(f"Rotation: ±{config.augmentation.rotation}°")
            logger.info(f"Zoom: {config.augmentation.zoom}")
            logger.info(f"Width shift: ±{config.augmentation.width_shift}")
            logger.info(f"Height shift: ±{config.augmentation.height_shift}")
            logger.info(f"Horizontal flip: {config.augmentation.horizontal_flip}")
        
        train_loader, test_loader = load_data()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = create_dynamic_model(config.network_architecture)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        
        if config.optimizer.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
            
        criterion = nn.CrossEntropyLoss().to(device)
        results = []
        
        for epoch in range(config.epochs):
            train_acc, train_loss = train_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                config.augmentation
            )
            test_acc, test_loss = test_epoch(model, test_loader, criterion)
            
            epoch_result = {
                "epoch": epoch,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_loss": train_loss,
                "test_loss": test_loss
            }
            results.append(epoch_result)
            
            logger.info(f"Epoch {epoch + 1} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        end_time = datetime.now()
        
        # Create new history entry with augmentation info
        new_entry = {
            "timestamp": start_time.isoformat(),
            "training_end_time": end_time.isoformat(),
            "architecture": [
                {
                    "type": layer.type,
                    "params": layer.params
                } for layer in config.network_architecture
            ],
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "total_params": total_params,
            "final_train_accuracy": train_acc,
            "final_test_accuracy": test_acc,
            "training_results": results,
            "training_time": (end_time - start_time).total_seconds(),
            "augmentation": config.augmentation.dict() if config.augmentation else None  # Add augmentation info
        }
        
        # Load existing history or create new
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            else:
                history = []
        except json.JSONDecodeError:
            history = []
        
        # Add new entry
        history.append(new_entry)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        
        # Save updated history
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Successfully saved training history to {HISTORY_FILE}")
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            logger.error(traceback.format_exc())
        
        return {
            "results": results,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def calculate_layer_params(layer_config, input_shape):
    if layer_config.type == 'Convolution 2D':
        in_channels = input_shape[0]
        kernel_size = layer_config.params['kernelSize']
        filters = layer_config.params['filters']
        # Parameters = (kernel_size * kernel_size * in_channels * filters) + filters(bias)
        params = (kernel_size * kernel_size * in_channels * filters) + filters
        # Calculate output shape
        h = ((input_shape[1] + 2*layer_config.params['padding'] - kernel_size) // 
             layer_config.params['stride'] + 1)
        w = ((input_shape[2] + 2*layer_config.params['padding'] - kernel_size) // 
             layer_config.params['stride'] + 1)
        output_shape = (filters, h, w)
        return params, output_shape
    
    elif layer_config.type == 'Max Pooling':
        # No parameters in pooling layer
        channels = input_shape[0]
        h = input_shape[1] // layer_config.params['stride']
        w = input_shape[2] // layer_config.params['stride']
        return 0, (channels, h, w)
    
    elif layer_config.type == 'Dropout':
        # No parameters in dropout layer
        return 0, input_shape
    
    elif layer_config.type == 'Fully Connected':
        if len(input_shape) > 1:
            # First FC layer after conv layers
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        else:
            # FC layer after another FC layer
            input_size = input_shape[0]
        units = layer_config.params['units']
        # Parameters = (input_size * units) + units(bias)
        params = (input_size * units) + units
        return params, (units,)

def create_dynamic_model(architecture):
    try:
        layers = []
        current_channels = 1  # MNIST input channels
        current_height = 28   # MNIST input height
        current_width = 28    # MNIST input width
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initial dimensions: channels={current_channels}, height={current_height}, width={current_width}")

        for layer in architecture:
            logger.info(f"\nProcessing layer: {layer.type}")
            logger.info(f"Input dimensions: channels={current_channels}, height={current_height}, width={current_width}")

            if layer.type == 'Convolution 2D':
                conv = nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=layer.params['filters'],
                    kernel_size=layer.params['kernelSize'],
                    stride=layer.params['stride'],
                    padding=layer.params['padding']
                )
                layers.append(conv)
                if layer.params['activation'] == 'relu':
                    layers.append(nn.ReLU())
                
                # Update dimensions
                current_height = ((current_height + 2*layer.params['padding'] - layer.params['kernelSize']) 
                                // layer.params['stride'] + 1)
                current_width = ((current_width + 2*layer.params['padding'] - layer.params['kernelSize']) 
                               // layer.params['stride'] + 1)
                current_channels = layer.params['filters']

            elif layer.type == 'Max Pooling':
                layers.append(nn.MaxPool2d(
                    kernel_size=layer.params['poolSize'],
                    stride=layer.params['stride']
                ))
                # Update dimensions
                current_height = current_height // layer.params['stride']
                current_width = current_width // layer.params['stride']

            elif layer.type == 'Global Average Pooling':
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                # After GAP, we have 1x1 spatial dimensions but same number of channels
                current_height = 1
                current_width = 1
                # channels remain the same
                logger.info(f"After Global Average Pooling: channels={current_channels}, size=1x1")
                # Add Flatten after GAP
                layers.append(nn.Flatten())
                # After flattening, all dimensions become a single channel
                input_size = current_channels
                current_channels = input_size
                logger.info(f"After Flatten: input_size={input_size}")

            elif layer.type == 'Flatten':
                layers.append(nn.Flatten())
                # Update dimensions
                current_channels = current_channels * current_height * current_width
                current_height = 1
                current_width = 1
                logger.info(f"After Flatten: input_size={current_channels}")

            elif layer.type == 'Dropout':
                layers.append(nn.Dropout(p=layer.params['rate']))

            elif layer.type == 'Fully Connected':
                input_size = current_channels
                if current_height > 1 and current_width > 1:
                    # If dimensions haven't been flattened yet
                    input_size = current_channels * current_height * current_width
                    layers.append(nn.Flatten())
                    logger.info(f"Added implicit Flatten: input_size={input_size}")

                logger.info(f"Creating Linear layer: input_size={input_size}, output_size={layer.params['units']}")
                layers.append(nn.Linear(input_size, layer.params['units']))
                
                if layer.params['activation'] == 'relu':
                    layers.append(nn.ReLU())
                elif layer.params['activation'] == 'softmax':
                    layers.append(nn.LogSoftmax(dim=1))
                
                current_channels = layer.params['units']
                current_height = 1
                current_width = 1

            logger.info(f"Output dimensions: channels={current_channels}, height={current_height}, width={current_width}")

        logger.info("\nFinal model summary:")
        logger.info(f"Number of layers: {len(layers)}")
        logger.info(f"Final output shape: ({current_channels}, {current_height}, {current_width})")
        
        model = nn.Sequential(*layers).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params}")
        
        # Print model structure
        logger.info("\nModel structure:")
        for idx, layer in enumerate(layers):
            logger.info(f"Layer {idx}: {layer}")
        
        return model
    except Exception as e:
        logger.error(f"Error creating dynamic model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.get("/model-params")
async def get_model_params(conv_layers: str):
    try:
        logger.info(f"Getting model parameters for layers: {conv_layers}")
        # Convert string of comma-separated numbers to list of integers
        layers = [int(x) for x in conv_layers.split(',')]
        if not layers:
            raise ValueError("No layers provided")
            
        model = create_model(layers)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {total_params} parameters")
        return {"total_params": total_params}
    except Exception as e:
        logger.error(f"Error getting model parameters: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 

@app.get("/history")
async def get_training_history():
    try:
        history = load_training_history()
        logger.info(f"Loaded training history: {history}")
        if not history:
            # Initialize empty history file if it doesn't exist
            with open(HISTORY_FILE, 'w') as f:
                json.dump([], f)
            return []
        return history
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set-default-architecture")
async def set_default_architecture(architecture: List[LayerConfig]):
    try:
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Attempting to save to: {DEFAULT_ARCHITECTURE_FILE}")
        
        # Create tests directory if it doesn't exist
        tests_dir = os.path.dirname(DEFAULT_ARCHITECTURE_FILE)
        logger.info(f"Creating tests directory if needed: {tests_dir}")
        os.makedirs(tests_dir, exist_ok=True)
        
        # Use absolute path for file operations
        with open(DEFAULT_ARCHITECTURE_FILE, 'w') as f:
            content = {
                "network_architecture": [
                    {
                        "type": layer.type,
                        "params": layer.params
                    } for layer in architecture
                ]
            }
            json.dump(content, f, indent=2)
            logger.info(f"Successfully wrote content to file")
            
        # Verify the file was written
        if os.path.exists(DEFAULT_ARCHITECTURE_FILE):
            with open(DEFAULT_ARCHITECTURE_FILE, 'r') as f:
                saved_content = json.load(f)
            logger.info(f"Verified saved content: {saved_content}")
            
        return {"message": "Default architecture saved successfully"}
    except Exception as e:
        logger.error(f"Error saving default architecture: {str(e)}")
        logger.error(f"Attempted to save to: {DEFAULT_ARCHITECTURE_FILE}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def ensure_valid_history_file():
    """Ensure the history file exists and is valid JSON"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupted, create new
                    with open(HISTORY_FILE, 'w') as f:
                        json.dump([], f)
        else:
            # Create new file if doesn't exist
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, 'w') as f:
                json.dump([], f)
    except Exception as e:
        logger.error(f"Error ensuring valid history file: {e}")
        logger.error(traceback.format_exc())

# Call this when the app starts
@app.on_event("startup")
async def startup_event():
    ensure_valid_history_file()

@app.post("/augmented-samples")
async def get_augmented_samples(config: dict):
    try:
        # Load a few sample images from MNIST
        train_loader, _ = load_data()
        samples = []
        for data, _ in train_loader:
            if len(samples) < 6:  # Get 6 samples
                samples.append(data[0])
            else:
                break

        augmented_images = []
        for img in samples:
            # Convert to PIL Image
            img_pil = TF.to_pil_image(img)
            
            # Apply augmentations based on config
            if config['enabled']:
                # Rotation
                if config['rotation'] > 0:
                    angle = random.uniform(-config['rotation'], config['rotation'])
                    img_pil = TF.rotate(img_pil, angle)
                
                # Zoom (scale)
                if config['zoom'] > 0:
                    scale = 1.0 + random.uniform(0, config['zoom'])
                    new_size = [int(dim * scale) for dim in img_pil.size]
                    img_pil = TF.resize(img_pil, new_size)
                    # Crop back to original size
                    i = (img_pil.size[0] - 28) // 2
                    j = (img_pil.size[1] - 28) // 2
                    img_pil = TF.crop(img_pil, j, i, 28, 28)
                
                # Shifts
                if config['width_shift'] > 0 or config['height_shift'] > 0:
                    width_shift = int(28 * random.uniform(-config['width_shift'], config['width_shift']))
                    height_shift = int(28 * random.uniform(-config['height_shift'], config['height_shift']))
                    img_pil = TF.affine(img_pil, 0, [width_shift, height_shift], 1, 0)
                
                # Horizontal flip
                if config['horizontal_flip'] and random.random() > 0.5:
                    img_pil = TF.hflip(img_pil)

            # Convert to base64
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            augmented_images.append(img_str)

        return {"images": augmented_images}
    except Exception as e:
        logger.error(f"Error generating augmented samples: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))