import React, { useState } from 'react';
import { Paper, Typography, Box, Card, Chip, Stack, Button } from '@mui/material';
import { AccessTime, Speed, Settings, Star, Transform } from '@mui/icons-material';
import { setDefaultArchitecture } from '../services/api';

const formatArchitecture = (architecture) => {
  return architecture.map((layer, index) => {
    switch (layer.type) {
      case 'Convolution 2D':
        return `Conv2D:${layer.params.filters}:${layer.params.activation}`;
      case 'Convolution 1x1':
        return `Conv1x1:${layer.params.filters}:${layer.params.activation}`;
      case 'Max Pooling':
        return 'MaxPool';
      case 'Global Average Pooling':
        return 'GlobalAvgPool';
      case 'Flatten':
        return 'Flatten';
      case 'Dropout':
        return `Drop:${layer.params.rate}`;
      case 'Batch Normalization':
        return 'BatchNorm';
      case 'Fully Connected':
        return `Dense:${layer.params.units}:${layer.params.activation}`;
      default:
        return layer.type;
    }
  }).join(' → ');
};

const formatDuration = (startTime, endTime) => {
  const duration = new Date(endTime) - new Date(startTime);
  const minutes = Math.floor(duration / 60000);
  const seconds = ((duration % 60000) / 1000).toFixed(0);
  return `${minutes}m ${seconds}s`;
};

const TrainingHistory = ({ history, onSetDefault }) => {
  const [selectedDefault, setSelectedDefault] = useState(() => {
    const saved = localStorage.getItem('defaultArchitecture');
    return saved ? JSON.stringify(JSON.parse(saved)) : null;
  });

  const handleSetDefault = async (entry) => {
    try {
      // Create the configuration object
      const configToSave = {
        network_architecture: entry.architecture,
        training_config: {
          optimizer: entry.optimizer,
          learning_rate: entry.learning_rate,
          epochs: entry.epochs,
          batch_size: entry.batch_size,  // Include batch size
          augmentation: entry.augmentation || {
            enabled: false,
            rotation: 0,
            zoom: 0,
            width_shift: 0,
            height_shift: 0,
            horizontal_flip: false
          }
        }
      };
      
      console.log('Setting default configuration:', configToSave);
      const configString = JSON.stringify(configToSave);
      
      // Save to localStorage
      localStorage.setItem('defaultArchitecture', configString);
      setSelectedDefault(configString);
      
      // Save to backend
      await setDefaultArchitecture(configToSave);
      
      if (onSetDefault) {
        onSetDefault(configToSave.network_architecture);
      }

      console.log('Successfully set default configuration:', configToSave);
    } catch (error) {
      console.error('Failed to set default configuration:', error);
    }
  };

  // Sort history by timestamp in descending order (latest first)
  const sortedHistory = [...history].sort((a, b) => {
    const dateA = new Date(a.timestamp);
    const dateB = new Date(b.timestamp);
    return dateB - dateA;
  });

  const formatAugmentationInfo = (augmentation) => {
    if (!augmentation || !augmentation.enabled) return null;
    
    const settings = [];
    if (augmentation.rotation > 0) settings.push(`Rot:±${augmentation.rotation}°`);
    if (augmentation.zoom > 0) settings.push(`Zoom:${augmentation.zoom}`);
    if (augmentation.width_shift > 0) settings.push(`W-Shift:${augmentation.width_shift}`);
    if (augmentation.height_shift > 0) settings.push(`H-Shift:${augmentation.height_shift}`);
    if (augmentation.horizontal_flip) settings.push('H-Flip');
    
    return settings.join(', ');
  };

  const isHighPerforming = (entry) => {
    return (
      entry.final_train_accuracy >= 0.95 &&
      entry.final_test_accuracy >= 0.95 &&
      entry.total_params < 10000
    );
  };

  return (
    <Paper sx={{ p: 2, maxHeight: '80vh', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        Training History
      </Typography>
      {sortedHistory.map((entry, index) => {
        const isDefault = selectedDefault === JSON.stringify(entry.architecture);
        const augmentationInfo = formatAugmentationInfo(entry.augmentation);
        const isOptimal = isHighPerforming(entry);
        
        return (
          <Card 
            key={entry.timestamp} 
            sx={{ 
              mb: 2, 
              p: 1,
              border: isOptimal ? '2px solid #4caf50' : 'none',
              backgroundColor: isOptimal ? '#f1f8e9' : 'white',
              position: 'relative'
            }}
          >
            {isOptimal && (
              <Box
                sx={{
                  position: 'absolute',
                  top: -10,
                  right: -10,
                  backgroundColor: '#4caf50',
                  color: 'white',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontSize: '0.75rem',
                  fontWeight: 'bold'
                }}
              >
                Optimal
              </Box>
            )}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="subtitle2" gutterBottom>
                Model #{sortedHistory.length - index} - {new Date(entry.timestamp).toLocaleString()}
              </Typography>
              <Button
                startIcon={<Star />}
                size="small"
                variant={isDefault ? "contained" : "outlined"}
                color="primary"
                onClick={() => handleSetDefault(entry)}
                title={isDefault ? "Current Default Architecture" : "Set as Default Architecture"}
                sx={{
                  backgroundColor: isDefault ? 'primary.main' : 'transparent',
                  color: isDefault ? 'white' : 'primary.main',
                  '&:hover': {
                    backgroundColor: isDefault ? 'primary.dark' : 'rgba(25, 118, 210, 0.04)',
                  }
                }}
              >
                {isDefault ? 'Default' : 'Set Default'}
              </Button>
            </Box>
            
            <Stack 
              direction="row" 
              spacing={1} 
              sx={{ mb: 1 }}
              flexWrap="wrap"  // Allow chips to wrap
              gap={1}  // Add gap between wrapped items
            >
              <Chip
                size="small"
                icon={<Settings />}
                label={`${entry.optimizer.toUpperCase()}`}
                color="primary"
                variant="outlined"
              />
              <Chip
                size="small"
                icon={<Speed />}
                label={`LR: ${entry.learning_rate}`}
                color="primary"
                variant="outlined"
              />
              <Chip
                size="small"
                icon={<AccessTime />}
                label={`${entry.epochs} epochs`}
                color="primary"
                variant="outlined"
              />
              <Chip
                size="small"
                icon={<Speed />}
                label={`Batch: ${entry.batch_size}`}
                color="primary"
                variant="outlined"
                sx={{ minWidth: 'auto' }}  // Allow chip to grow as needed
              />
              {augmentationInfo && (
                <Chip
                  size="small"
                  icon={<Transform />}
                  label="Augmented"
                  color="secondary"
                  variant="outlined"
                  title={augmentationInfo}
                />
              )}
            </Stack>

            <Typography 
              variant="caption" 
              color="text.secondary" 
              sx={{ 
                display: 'block', 
                mb: 1,
                whiteSpace: 'normal',
                wordBreak: 'break-word'
              }}
            >
              {formatArchitecture(entry.architecture)}
              {augmentationInfo && (
                <Box component="span" sx={{ ml: 1, color: 'secondary.main' }}>
                  ({augmentationInfo})
                </Box>
              )}
            </Typography>

            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 1
            }}>
              <Box>
                <Typography 
                  variant="body2" 
                  color={isOptimal ? 'success.main' : 'text.secondary'}
                  fontWeight={isOptimal ? 'bold' : 'normal'}
                >
                  Train: {(entry.final_train_accuracy * 100).toFixed(2)}% | 
                  Test: {(entry.final_test_accuracy * 100).toFixed(2)}%
                </Typography>
                <Typography 
                  variant="caption" 
                  color={isOptimal ? 'success.main' : 'text.secondary'}
                  fontWeight={isOptimal ? 'bold' : 'normal'}
                >
                  Params: {entry.total_params.toLocaleString()}
                </Typography>
              </Box>
              {entry.training_time && (
                <Typography variant="caption" color="text.secondary">
                  Time: {formatDuration(entry.timestamp, entry.training_end_time)}
                </Typography>
              )}
            </Box>
          </Card>
        );
      })}
      {history.length === 0 && (
        <Typography variant="body2" color="text.secondary" textAlign="center" py={4}>
          No training history yet
        </Typography>
      )}
    </Paper>
  );
};

export default TrainingHistory; 