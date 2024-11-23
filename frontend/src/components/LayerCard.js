import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  IconButton,
  Collapse,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Popover
} from '@mui/material';
import { Delete, ExpandMore, ExpandLess, Calculate } from '@mui/icons-material';
import LayerVisualization from './LayerVisualization';

const LayerCard = ({ layer, isTemplate, onDelete, onUpdate, dimensions, previousLayers }) => {
  const [expanded, setExpanded] = useState(false);
  const [params, setParams] = useState(layer?.defaultParams || {});
  const [anchorEl, setAnchorEl] = useState(null);

  useEffect(() => {
    if (layer?.defaultParams) {
      setParams(layer.defaultParams);
    }
  }, [layer]);

  const handleDelete = () => {
    setAnchorEl(null);
    setExpanded(false);
    if (onDelete) {
      onDelete();
    }
  };

  const handleParamChange = (param, value) => {
    if (!layer) return;
    
    const newParams = { ...params, [param]: value };
    setParams(newParams);
    if (onUpdate) {
      onUpdate(newParams);
    }
  };

  const handleParamClick = (event) => {
    if (!isTemplate && dimensions) {
      setAnchorEl(event.currentTarget);
    }
  };

  const handlePopoverClose = () => {
    setAnchorEl(null);
  };

  const renderDimensions = () => {
    if (isTemplate || !dimensions) return null;

    return (
      <Box 
        component="span" 
        sx={{ 
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          color: 'text.secondary',
          fontSize: '0.75rem',
          ml: 1
        }}
      >
        ({dimensions.input.join('×')} → {dimensions.output.join('×')})
        <Box 
          component="span" 
          sx={{ 
            display: 'flex',
            alignItems: 'center',
            cursor: dimensions.paramBreakdown ? 'pointer' : 'default',
          }}
          onClick={handleParamClick}
        >
          [{dimensions.params.toLocaleString()}p
          {dimensions.paramBreakdown && <Calculate fontSize="small" sx={{ ml: 0.5 }} />}]
        </Box>
      </Box>
    );
  };

  const renderParams = () => {
    if (!layer?.type) return null;

    switch (layer.type) {
      case 'Convolution 2D':
        return (
          <>
            <TextField
              size="small"
              label="Filters"
              type="number"
              value={params.filters}
              onChange={(e) => handleParamChange('filters', parseInt(e.target.value))}
              sx={{ mb: 1 }}
              fullWidth
            />
            <TextField
              size="small"
              label="Kernel Size"
              type="number"
              value={params.kernelSize}
              onChange={(e) => handleParamChange('kernelSize', parseInt(e.target.value))}
              sx={{ mb: 1 }}
              fullWidth
            />
            <TextField
              size="small"
              label="Stride"
              type="number"
              value={params.stride}
              onChange={(e) => handleParamChange('stride', parseInt(e.target.value))}
              sx={{ mb: 1 }}
              fullWidth
            />
            <TextField
              size="small"
              label="Padding"
              type="number"
              value={params.padding}
              onChange={(e) => handleParamChange('padding', parseInt(e.target.value))}
              sx={{ mb: 1 }}
              fullWidth
            />
            <FormControl fullWidth size="small">
              <InputLabel>Activation</InputLabel>
              <Select
                value={params.activation}
                onChange={(e) => handleParamChange('activation', e.target.value)}
                label="Activation"
              >
                <MenuItem value="relu">ReLU</MenuItem>
                <MenuItem value="tanh">Tanh</MenuItem>
                <MenuItem value="sigmoid">Sigmoid</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      case 'Convolution 1x1':
        return (
          <>
            <TextField
              size="small"
              label="Filters"
              type="number"
              value={params.filters}
              onChange={(e) => handleParamChange('filters', parseInt(e.target.value))}
              sx={{ mb: 1 }}
              fullWidth
            />
            <FormControl fullWidth size="small">
              <InputLabel>Activation</InputLabel>
              <Select
                value={params.activation}
                onChange={(e) => handleParamChange('activation', e.target.value)}
                label="Activation"
              >
                <MenuItem value="relu">ReLU</MenuItem>
                <MenuItem value="tanh">Tanh</MenuItem>
                <MenuItem value="sigmoid">Sigmoid</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      case 'Batch Normalization':
        return (
          <>
            <TextField
              size="small"
              label="Momentum"
              type="number"
              value={params.momentum}
              onChange={(e) => handleParamChange('momentum', parseFloat(e.target.value))}
              sx={{ mb: 1 }}
              fullWidth
              inputProps={{ step: 0.01, min: 0, max: 1 }}
            />
            <TextField
              size="small"
              label="Epsilon"
              type="number"
              value={params.epsilon}
              onChange={(e) => handleParamChange('epsilon', parseFloat(e.target.value))}
              fullWidth
              inputProps={{ step: 0.001, min: 0.001 }}
            />
          </>
        );
      case 'Max Pooling':
        return (
          <>
            <TextField
              size="small"
              label="Pool Size"
              type="number"
              value={params.poolSize}
              onChange={(e) => handleParamChange('poolSize', parseInt(e.target.value))}
              sx={{ mb: 1 }}
              fullWidth
            />
            <TextField
              size="small"
              label="Stride"
              type="number"
              value={params.stride}
              onChange={(e) => handleParamChange('stride', parseInt(e.target.value))}
              fullWidth
            />
          </>
        );
      case 'Dropout':
        return (
          <TextField
            size="small"
            label="Rate"
            type="number"
            value={params.rate}
            onChange={(e) => handleParamChange('rate', parseFloat(e.target.value))}
            fullWidth
            inputProps={{ step: 0.1, min: 0, max: 1 }}
          />
        );
      case 'Fully Connected':
        return (
          <>
            <TextField
              size="small"
              label="Units"
              type="number"
              value={params.units}
              onChange={(e) => handleParamChange('units', parseInt(e.target.value))}
              sx={{ mb: 1 }}
              fullWidth
            />
            <FormControl fullWidth size="small">
              <InputLabel>Activation</InputLabel>
              <Select
                value={params.activation}
                onChange={(e) => handleParamChange('activation', e.target.value)}
                label="Activation"
              >
                <MenuItem value="relu">ReLU</MenuItem>
                <MenuItem value="softmax">Softmax</MenuItem>
                <MenuItem value="sigmoid">Sigmoid</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      default:
        return null;
    }
  };

  const open = Boolean(anchorEl);

  if (!layer || !layer.type) return null;

  return (
    <Paper
      sx={{
        p: 0.75,
        mb: 0.5,
        width: isTemplate ? '120px' : '100%',
        backgroundColor: isTemplate ? '#f5f5f5' : 'white',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', overflow: 'hidden' }}>
          <Typography 
            variant="body2" 
            sx={{ 
              fontWeight: 500,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}
          >
            {layer.icon} {getShortLayerName(layer.type)}
            {renderDimensions()}
          </Typography>
        </Box>
        {!isTemplate && (
          <Box sx={{ ml: 1, display: 'flex', alignItems: 'center' }}>
            <LayerVisualization 
              layer={layer} 
              dimensions={dimensions}
              previousLayers={previousLayers}
            />
            <IconButton 
              size="small" 
              onClick={() => setExpanded(!expanded)}
              disabled={!layer}
            >
              {expanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
            <IconButton 
              size="small" 
              onClick={handleDelete}
              disabled={!layer}
            >
              <Delete />
            </IconButton>
          </Box>
        )}
      </Box>
      {!isTemplate && layer && (
        <Collapse in={expanded}>
          <Box sx={{ mt: 1 }}>
            {renderParams()}
          </Box>
        </Collapse>
      )}
      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handlePopoverClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
      >
        <Box sx={{ p: 1.5, maxWidth: 300 }}>
          <Typography variant="caption">
            {dimensions?.paramBreakdown}
          </Typography>
        </Box>
      </Popover>
    </Paper>
  );
};

// Helper function to shorten layer names
const getShortLayerName = (type) => {
  switch (type) {
    case 'Convolution 2D':
      return 'Conv2D';
    case 'Convolution 1x1':
      return 'Conv1x1';
    case 'Max Pooling':
      return 'MaxPool';
    case 'Global Average Pooling':
      return 'GAvgPool';
    case 'Batch Normalization':
      return 'BatchNorm';
    case 'Fully Connected':
      return 'Dense';
    default:
      return type;
  }
};

export default LayerCard; 