import React, { useEffect, useState } from 'react';
import { Paper, Typography, Box } from '@mui/material';
import LayerCard from './LayerCard';

const calculateLayerDimensions = (layers) => {
  let dimensions = [];
  let inputChannels = 1;  // MNIST input channels
  let currentHeight = 28; // MNIST input height
  let currentWidth = 28;  // MNIST input width

  layers.forEach((layer, index) => {
    let inputDim = [inputChannels, currentHeight, currentWidth];
    let outputDim = [...inputDim];
    let params = 0;

    if (layer.type === 'Convolution 2D') {
      // Update output dimensions
      outputDim[0] = layer.defaultParams.filters;
      outputDim[1] = Math.floor(
        (currentHeight + 2*layer.defaultParams.padding - layer.defaultParams.kernelSize) / 
        layer.defaultParams.stride + 1
      );
      outputDim[2] = Math.floor(
        (currentWidth + 2*layer.defaultParams.padding - layer.defaultParams.kernelSize) / 
        layer.defaultParams.stride + 1
      );
      
      // Calculate parameters
      params = (layer.defaultParams.kernelSize * layer.defaultParams.kernelSize * 
                inputChannels * layer.defaultParams.filters) + layer.defaultParams.filters;
      
      // Update for next layer
      inputChannels = layer.defaultParams.filters;
      currentHeight = outputDim[1];
      currentWidth = outputDim[2];
    } 
    else if (layer.type === 'Max Pooling') {
      outputDim[1] = Math.floor(currentHeight / layer.defaultParams.stride);
      outputDim[2] = Math.floor(currentWidth / layer.defaultParams.stride);
      
      currentHeight = outputDim[1];
      currentWidth = outputDim[2];
    }
    else if (layer.type === 'Global Average Pooling') {
      outputDim = [inputChannels, 1, 1];
      currentHeight = 1;
      currentWidth = 1;
      // No parameters in global average pooling
      params = 0;
    }
    else if (layer.type === 'Flatten') {
      outputDim = [inputChannels * currentHeight * currentWidth];
      currentHeight = 1;
      currentWidth = 1;
      inputChannels = outputDim[0];
    }
    else if (layer.type === 'Dropout') {
      // Dimensions remain the same
      outputDim = [...inputDim];
    }
    else if (layer.type === 'Fully Connected') {
      let inputSize;
      if (currentHeight > 1 || currentWidth > 1) {
        inputSize = inputChannels * currentHeight * currentWidth;
        inputDim = [inputSize];
      } else {
        inputSize = inputChannels;
      }
      outputDim = [layer.defaultParams.units];
      params = (inputSize * layer.defaultParams.units) + layer.defaultParams.units;
      
      inputChannels = layer.defaultParams.units;
      currentHeight = 1;
      currentWidth = 1;
    }

    dimensions.push({
      input: inputDim,
      output: outputDim,
      params: params
    });
  });

  return dimensions;
};

const NetworkBuilder = ({ layers, onLayerUpdate, onLayerDelete }) => {
  const [totalParams, setTotalParams] = useState(0);
  const [layerDimensions, setLayerDimensions] = useState([]);

  useEffect(() => {
    const dimensions = calculateLayerDimensions(layers);
    setLayerDimensions(dimensions);
    setTotalParams(dimensions.reduce((sum, dim) => sum + dim.params, 0));
  }, [layers]);

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Network Architecture
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Total Parameters: {totalParams.toLocaleString()}
        </Typography>
      </Box>
      <Box sx={{ minHeight: 200, p: 2, backgroundColor: '#f8f8f8' }}>
        {layers.map((layer, index) => (
          <LayerCard
            key={layer.id}
            layer={layer}
            onDelete={() => onLayerDelete(index)}
            onUpdate={(params) => onLayerUpdate(index, params)}
            dimensions={layerDimensions[index]}
          />
        ))}
        {layers.length === 0 && (
          <Typography variant="body2" color="text.secondary" textAlign="center" py={4}>
            Drag and drop layers here to build your network
          </Typography>
        )}
      </Box>
    </Paper>
  );
};

export default NetworkBuilder; 