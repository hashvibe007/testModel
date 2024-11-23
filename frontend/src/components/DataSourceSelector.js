import React from 'react';
import { 
  Paper, 
  Typography, 
  RadioGroup, 
  FormControlLabel, 
  Radio, 
  Box,
  Tooltip
} from '@mui/material';

const dataSources = [
  {
    id: 'mnist',
    name: 'MNIST',
    description: 'Handwritten digits (28x28, grayscale)',
    dimensions: [1, 28, 28]  // [channels, height, width]
  },
  {
    id: 'cifar10',
    name: 'CIFAR-10',
    description: 'Real-world objects (32x32, RGB)',
    dimensions: [3, 32, 32],
    disabled: true
  },
  {
    id: 'fashion_mnist',
    name: 'Fashion MNIST',
    description: 'Fashion items (28x28, grayscale)',
    dimensions: [1, 28, 28],
    disabled: true
  }
];

const DataSourceSelector = ({ selectedSource, onSourceChange }) => {
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        Data Source
      </Typography>
      <RadioGroup
        value={selectedSource}
        onChange={(e) => onSourceChange(e.target.value)}
      >
        {dataSources.map((source) => (
          <Tooltip 
            key={source.id}
            title={source.disabled ? "Coming soon!" : source.description}
            placement="right"
          >
            <Box>
              <FormControlLabel
                value={source.id}
                control={<Radio size="small" />}
                label={
                  <Box>
                    <Typography variant="body2">
                      {source.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Input: {source.dimensions.join(' × ')}
                    </Typography>
                  </Box>
                }
                disabled={source.disabled}
                sx={{ mb: 1 }}
              />
            </Box>
          </Tooltip>
        ))}
      </RadioGroup>
    </Paper>
  );
};

export default DataSourceSelector; 