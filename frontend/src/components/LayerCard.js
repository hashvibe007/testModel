import React, { useState } from 'react';
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
  Grid
} from '@mui/material';
import { Delete, ExpandMore, ExpandLess } from '@mui/icons-material';

const LayerCard = ({ layer, isTemplate, onDelete, onUpdate, dimensions }) => {
  const [expanded, setExpanded] = useState(false);
  const [params, setParams] = useState(layer.defaultParams);

  const handleParamChange = (param, value) => {
    const newParams = { ...params, [param]: value };
    setParams(newParams);
    if (onUpdate) {
      onUpdate(newParams);
    }
  };

  const renderDimensions = () => {
    if (isTemplate || !dimensions) return null;

    return (
      <Box sx={{ mt: 1, p: 1, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
        <Grid container spacing={1}>
          <Grid item xs={4}>
            <Typography variant="caption" display="block">
              Input: {dimensions.input.join(' × ')}
            </Typography>
          </Grid>
          <Grid item xs={4}>
            <Typography variant="caption" display="block">
              Output: {dimensions.output.join(' × ')}
            </Typography>
          </Grid>
          <Grid item xs={4}>
            <Typography variant="caption" display="block">
              Params: {dimensions.params.toLocaleString()}
            </Typography>
          </Grid>
        </Grid>
      </Box>
    );
  };

  const renderParams = () => {
    switch (layer.type) {
      case 'Convolution 2D':
        return (
          <>
            <TextField
              fullWidth
              label="Filters"
              type="number"
              value={params.filters}
              onChange={(e) => handleParamChange('filters', parseInt(e.target.value))}
              sx={{ mb: 1 }}
            />
            <TextField
              fullWidth
              label="Kernel Size"
              type="number"
              value={params.kernelSize}
              onChange={(e) => handleParamChange('kernelSize', parseInt(e.target.value))}
              sx={{ mb: 1 }}
            />
            <TextField
              fullWidth
              label="Stride"
              type="number"
              value={params.stride}
              onChange={(e) => handleParamChange('stride', parseInt(e.target.value))}
              sx={{ mb: 1 }}
            />
            <TextField
              fullWidth
              label="Padding"
              type="number"
              value={params.padding}
              onChange={(e) => handleParamChange('padding', parseInt(e.target.value))}
              sx={{ mb: 1 }}
            />
            <FormControl fullWidth>
              <InputLabel>Activation</InputLabel>
              <Select
                value={params.activation}
                onChange={(e) => handleParamChange('activation', e.target.value)}
              >
                <MenuItem value="relu">ReLU</MenuItem>
                <MenuItem value="tanh">Tanh</MenuItem>
                <MenuItem value="sigmoid">Sigmoid</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      case 'Max Pooling':
        return (
          <>
            <TextField
              fullWidth
              label="Pool Size"
              type="number"
              value={params.poolSize}
              onChange={(e) => handleParamChange('poolSize', parseInt(e.target.value))}
              sx={{ mb: 1 }}
            />
            <TextField
              fullWidth
              label="Stride"
              type="number"
              value={params.stride}
              onChange={(e) => handleParamChange('stride', parseInt(e.target.value))}
            />
          </>
        );
      case 'Dropout':
        return (
          <TextField
            fullWidth
            label="Rate"
            type="number"
            value={params.rate}
            onChange={(e) => handleParamChange('rate', parseFloat(e.target.value))}
            inputProps={{ step: 0.1, min: 0, max: 1 }}
          />
        );
      case 'Fully Connected':
        return (
          <>
            <TextField
              fullWidth
              label="Units"
              type="number"
              value={params.units}
              onChange={(e) => handleParamChange('units', parseInt(e.target.value))}
              sx={{ mb: 1 }}
            />
            <FormControl fullWidth>
              <InputLabel>Activation</InputLabel>
              <Select
                value={params.activation}
                onChange={(e) => handleParamChange('activation', e.target.value)}
              >
                <MenuItem value="relu">ReLU</MenuItem>
                <MenuItem value="tanh">Tanh</MenuItem>
                <MenuItem value="sigmoid">Sigmoid</MenuItem>
                <MenuItem value="softmax">Softmax</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      default:
        return null;
    }
  };

  return (
    <Paper
      sx={{
        p: 1,
        mb: 1,
        width: isTemplate ? '120px' : '100%',
        backgroundColor: isTemplate ? '#f5f5f5' : 'white',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="subtitle2">
          {layer.icon} {layer.type}
        </Typography>
        <Box>
          {!isTemplate && (
            <>
              <IconButton size="small" onClick={() => setExpanded(!expanded)}>
                {expanded ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
              <IconButton size="small" onClick={onDelete}>
                <Delete />
              </IconButton>
            </>
          )}
        </Box>
      </Box>
      {renderDimensions()}
      {!isTemplate && (
        <Collapse in={expanded}>
          <Box sx={{ mt: 1 }}>
            {renderParams()}
          </Box>
        </Collapse>
      )}
    </Paper>
  );
};

export default LayerCard; 