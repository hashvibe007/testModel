import React from 'react';
import { Paper, Typography, Box } from '@mui/material';
import LayerCard from './LayerCard';

const layerTypes = [
  {
    id: 'conv2d',
    type: 'Convolution 2D',
    icon: '⚡',
    defaultParams: {
      filters: 32,
      kernelSize: 3,
      stride: 1,
      padding: 1,
      activation: 'relu'
    }
  },
  {
    id: 'maxpool',
    type: 'Max Pooling',
    icon: '🔲',
    defaultParams: {
      poolSize: 2,
      stride: 2
    }
  },
  {
    id: 'globalavgpool',
    type: 'Global Average Pooling',
    icon: '🌐',
    defaultParams: {}
  },
  {
    id: 'flatten',
    type: 'Flatten',
    icon: '📄',
    defaultParams: {}
  },
  {
    id: 'dropout',
    type: 'Dropout',
    icon: '💧',
    defaultParams: {
      rate: 0.25
    }
  },
  {
    id: 'dense',
    type: 'Fully Connected',
    icon: '🔌',
    defaultParams: {
      units: 128,
      activation: 'relu'
    }
  }
];

const LayerPalette = () => {
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        Available Layers
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
        {layerTypes.map((layer) => (
          <Box
            key={layer.id}
            sx={{
              cursor: 'grab',
              '&:active': { cursor: 'grabbing' }
            }}
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('layerType', layer.id);
            }}
          >
            <LayerCard layer={layer} isTemplate={true} />
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default LayerPalette; 