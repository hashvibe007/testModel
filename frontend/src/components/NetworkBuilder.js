import React, { useEffect, useState } from 'react';
import { Paper, Typography, Box, Grid, Popover } from '@mui/material';
import LayerCard from './LayerCard';
import { Calculate } from '@mui/icons-material';

const calculateLayerDimensions = (layers) => {
  let dimensions = [];
  let inputChannels = 1;  // MNIST input channels
  let currentHeight = 28; // MNIST input height
  let currentWidth = 28;  // MNIST input width

  layers.forEach((layer, index) => {
    let inputDim = [inputChannels, currentHeight, currentWidth];
    let outputDim = [...inputDim];
    let params = 0;
    let paramBreakdown = '';  // Add parameter breakdown explanation

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
      
      // Calculate parameters with breakdown
      const kernelParams = layer.defaultParams.kernelSize * layer.defaultParams.kernelSize * 
                          inputChannels * layer.defaultParams.filters;
      const biasParams = layer.defaultParams.filters;
      params = kernelParams + biasParams;
      
      paramBreakdown = `Kernel: (${layer.defaultParams.kernelSize}×${layer.defaultParams.kernelSize}×${inputChannels}×${layer.defaultParams.filters}) + Bias: ${biasParams}`;
      
      // Update for next layer
      inputChannels = layer.defaultParams.filters;
      currentHeight = outputDim[1];
      currentWidth = outputDim[2];
    } 
    else if (layer.type === 'Convolution 1x1') {
      outputDim[0] = layer.defaultParams.filters;
      const kernelParams = 1 * 1 * inputChannels * layer.defaultParams.filters;
      const biasParams = layer.defaultParams.filters;
      params = kernelParams + biasParams;
      
      paramBreakdown = `Kernel: (1×1×${inputChannels}×${layer.defaultParams.filters}) + Bias: ${biasParams}`;
      
      inputChannels = layer.defaultParams.filters;
    }
    else if (layer.type === 'Batch Normalization') {
      outputDim = [...inputDim];
      params = 4 * inputChannels;  // gamma, beta, running_mean, running_var
      paramBreakdown = `4 params per channel × ${inputChannels} channels`;
    }
    else if (layer.type === 'Max Pooling') {
      outputDim[1] = Math.floor(currentHeight / layer.defaultParams.stride);
      outputDim[2] = Math.floor(currentWidth / layer.defaultParams.stride);
      paramBreakdown = 'No parameters';
      currentHeight = outputDim[1];
      currentWidth = outputDim[2];
    }
    else if (layer.type === 'Global Average Pooling') {
      outputDim = [inputChannels, 1, 1];
      currentHeight = 1;
      currentWidth = 1;
      paramBreakdown = 'No parameters';
    }
    else if (layer.type === 'Flatten') {
      outputDim = [inputChannels * currentHeight * currentWidth];
      currentHeight = 1;
      currentWidth = 1;
      inputChannels = outputDim[0];
      paramBreakdown = 'No parameters';
    }
    else if (layer.type === 'Dropout') {
      outputDim = [...inputDim];
      paramBreakdown = 'No parameters';
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
      const weightParams = inputSize * layer.defaultParams.units;
      const biasParams = layer.defaultParams.units;
      params = weightParams + biasParams;
      
      paramBreakdown = `Weights: (${inputSize}×${layer.defaultParams.units}) + Bias: ${biasParams}`;
      
      inputChannels = layer.defaultParams.units;
      currentHeight = 1;
      currentWidth = 1;
    }

    dimensions.push({
      input: inputDim,
      output: outputDim,
      params: params,
      paramBreakdown: paramBreakdown  // Add breakdown to dimensions
    });
  });

  return dimensions;
};

const renderDimensions = (dimensions) => {
  if (!dimensions) return null;

  return (
    <Box sx={{ mt: 1, p: 1, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
      <Grid container spacing={1}>
        <Grid item xs={12}>
          <Typography variant="caption" display="block">
            Input: {dimensions.input.join(' × ')}
          </Typography>
        </Grid>
        <Grid item xs={12}>
          <Typography variant="caption" display="block">
            Output: {dimensions.output.join(' × ')}
          </Typography>
        </Grid>
        <Grid item xs={12}>
          <Typography variant="caption" display="block">
            Parameters: {dimensions.params.toLocaleString()}
            {dimensions.paramBreakdown && (
              <Box component="span" sx={{ display: 'block', color: 'text.secondary' }}>
                ({dimensions.paramBreakdown})
              </Box>
            )}
          </Typography>
        </Grid>
      </Grid>
    </Box>
  );
};

const NetworkBuilder = ({ layers, onLayerUpdate, onLayerDelete, layerTypes }) => {
  const [totalParams, setTotalParams] = useState(0);
  const [layerDimensions, setLayerDimensions] = useState([]);
  const [dropTarget, setDropTarget] = useState(null);
  const [anchorEl, setAnchorEl] = useState(null);

  useEffect(() => {
    const dimensions = calculateLayerDimensions(layers);
    setLayerDimensions(dimensions);
    setTotalParams(dimensions.reduce((sum, dim) => sum + dim.params, 0));
  }, [layers]);

  const handleDragOver = (e, index) => {
    e.preventDefault();
    setDropTarget(index);
  };

  const handleDrop = (e, index) => {
    e.preventDefault();
    const layerType = e.dataTransfer.getData('layerType');
    if (!layerType) return;

    const layerTemplate = layerTypes[layerType];
    
    if (layerTemplate) {
      const newLayer = {
        id: `${layerType}-${Date.now()}`,
        type: layerTemplate.type,
        icon: layerTemplate.icon,
        defaultParams: { ...layerTemplate.defaultParams }
      };
      
      // Create new array with layer inserted at drop position
      const newLayers = [...layers];
      newLayers.splice(index, 0, newLayer);
      
      // Validate layer ordering
      if (validateLayerOrder(newLayers)) {
        onLayerUpdate(newLayers);
      }
    }
    
    setDropTarget(null);
  };

  const validateLayerOrder = (newLayers) => {
    let hasSeenFC = false;
    let hasSeenFlatten = false;

    for (const layer of newLayers) {
      // Once we see a FC layer, we can't have conv or pooling layers after it
      if (hasSeenFC && (
        layer.type === 'Convolution 2D' || 
        layer.type === 'Convolution 1x1' || 
        layer.type === 'Max Pooling' || 
        layer.type === 'Global Average Pooling'
      )) {
        return false;
      }

      // Once we see a flatten layer, we can only have FC or dropout layers after it
      if (hasSeenFlatten && (
        layer.type !== 'Fully Connected' && 
        layer.type !== 'Dropout'
      )) {
        return false;
      }

      if (layer.type === 'Fully Connected') {
        hasSeenFC = true;
      }

      if (layer.type === 'Flatten' || layer.type === 'Global Average Pooling') {
        hasSeenFlatten = true;
      }
    }

    return true;
  };

  const handleLayerDelete = (index) => {
    // Create a new array without the deleted layer
    const newLayers = layers.filter((_, i) => i !== index);
    onLayerDelete(index);
    
    // Recalculate dimensions for remaining layers
    const newDimensions = calculateLayerDimensions(newLayers);
    setLayerDimensions(newDimensions);
    setTotalParams(newDimensions.reduce((sum, dim) => sum + dim.params, 0));
  };

  const handleTotalParamsClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handlePopoverClose = () => {
    setAnchorEl(null);
  };

  const open = Boolean(anchorEl);

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Network Architecture
        </Typography>
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            cursor: 'pointer' 
          }}
          onClick={handleTotalParamsClick}
        >
          <Typography variant="subtitle1" color="text.secondary" sx={{ mr: 0.5 }}>
            Total Parameters: {totalParams.toLocaleString()}
          </Typography>
          <Calculate fontSize="small" color="action" />
        </Box>
      </Box>
      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handlePopoverClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <Box sx={{ p: 2, maxWidth: 400 }}>
          <Typography variant="subtitle2" gutterBottom>
            Parameter Breakdown by Layer:
          </Typography>
          {layerDimensions.map((dim, index) => (
            <Box key={index} sx={{ mb: 1 }}>
              <Typography variant="body2" color="text.secondary">
                Layer {index + 1} ({layers[index].type}):
              </Typography>
              <Typography variant="body2">
                {dim.paramBreakdown}
              </Typography>
            </Box>
          ))}
        </Box>
      </Popover>
      <Box sx={{ minHeight: 200, p: 2, backgroundColor: '#f8f8f8', borderRadius: 1 }}>
        {/* Drop zone for first position */}
        <Box
          onDragOver={(e) => handleDragOver(e, 0)}
          onDrop={(e) => handleDrop(e, 0)}
          sx={{
            height: '8px',
            backgroundColor: dropTarget === 0 ? '#bbdefb' : 'transparent',
            transition: 'background-color 0.2s',
            borderRadius: 1
          }}
        />
        
        {layers.map((layer, index) => (
          <Box key={layer.id}>
            <LayerCard
              layer={layer}
              onDelete={() => handleLayerDelete(index)}
              onUpdate={(params) => {
                const newLayers = [...layers];
                newLayers[index] = { ...layer, defaultParams: params };
                onLayerUpdate(newLayers);
              }}
              dimensions={layerDimensions[index]}
              previousLayers={layers.slice(0, index)}
            />
            {/* Drop zone after each layer */}
            <Box
              onDragOver={(e) => handleDragOver(e, index + 1)}
              onDrop={(e) => handleDrop(e, index + 1)}
              sx={{
                height: '8px',
                backgroundColor: dropTarget === index + 1 ? '#bbdefb' : 'transparent',
                transition: 'background-color 0.2s',
                borderRadius: 1
              }}
            />
          </Box>
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