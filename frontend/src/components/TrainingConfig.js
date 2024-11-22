import React, { useState } from 'react';
import {
  Paper,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Typography,
  Alert
} from '@mui/material';
import { trainModel } from '../services/api';

const TrainingConfig = ({ networkLayers, setTrainingResults, isTraining, setIsTraining, onTrainingComplete }) => {
  const [optimizer, setOptimizer] = useState('adam');
  const [learningRate, setLearningRate] = useState(0.001);
  const [epochs, setEpochs] = useState(10);
  const [error, setError] = useState(null);

  const validateNetwork = () => {
    if (networkLayers.length === 0) {
      return "Please add at least one layer to the network";
    }
    
    // Check layer ordering
    let hasSeenFC = false;
    for (let i = 0; i < networkLayers.length; i++) {
      const layer = networkLayers[i];
      
      // Once we see a FC layer, we can't have conv or pooling layers after it
      if (hasSeenFC && (layer.type === 'Convolution 2D' || layer.type === 'Max Pooling')) {
        return "Convolutional and Pooling layers must come before Fully Connected layers";
      }
      
      if (layer.type === 'Fully Connected') {
        hasSeenFC = true;
      }
    }
    
    // Check if the last layer is Dense with appropriate units for MNIST (10 classes)
    const lastLayer = networkLayers[networkLayers.length - 1];
    if (lastLayer.type !== 'Fully Connected' || lastLayer.defaultParams.units !== 10) {
      return "The last layer must be a Fully Connected layer with 10 units for MNIST classification";
    }

    return null;
  };

  const handleSubmit = async () => {
    try {
      const networkError = validateNetwork();
      if (networkError) {
        setError(networkError);
        return;
      }

      setIsTraining(true);
      setError(null);

      const config = {
        network_architecture: networkLayers.map(layer => ({
          type: layer.type,
          params: layer.defaultParams
        })),
        optimizer,
        learning_rate: learningRate,
        epochs
      };

      const data = await trainModel(config);
      setTrainingResults(data.results);
      if (onTrainingComplete) {
        await onTrainingComplete();
      }
      setError(null);
    } catch (err) {
      setError("Training failed: " + err.message);
      console.error('Training failed:', err);
    } finally {
      setIsTraining(false);
    }
  };

  // Clear error when parameters change
  const handleParameterChange = (paramSetter) => (event) => {
    setError(null);
    paramSetter(event.target.value);
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Training Configuration
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Optimizer</InputLabel>
        <Select
          value={optimizer}
          onChange={handleParameterChange(setOptimizer)}
          label="Optimizer"
        >
          <MenuItem value="adam">Adam</MenuItem>
          <MenuItem value="sgd">SGD</MenuItem>
        </Select>
      </FormControl>

      <TextField
        fullWidth
        label="Learning Rate"
        type="number"
        value={learningRate}
        onChange={handleParameterChange((value) => setLearningRate(parseFloat(value)))}
        sx={{ mb: 2 }}
        inputProps={{ min: 0.0001, step: 0.0001 }}
      />

      <TextField
        fullWidth
        label="Epochs"
        type="number"
        value={epochs}
        onChange={handleParameterChange((value) => setEpochs(parseInt(value)))}
        sx={{ mb: 2 }}
        inputProps={{ min: 1 }}
      />

      <Button
        variant="contained"
        onClick={handleSubmit}
        disabled={isTraining || networkLayers.length === 0}  // Only disable during training or when no layers
        fullWidth
      >
        {isTraining ? 'Training...' : 'Start Training'}
      </Button>
    </Paper>
  );
};

export default TrainingConfig; 