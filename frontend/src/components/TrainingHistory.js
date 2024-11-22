import React, { useState } from 'react';
import { Paper, Typography, Box, Card, CardContent, Chip, Stack, Button } from '@mui/material';
import { AccessTime, Speed, Settings, Star } from '@mui/icons-material';
import { setDefaultArchitecture } from '../services/api';

const formatArchitecture = (architecture) => {
  return architecture.map((layer, index) => {
    switch (layer.type) {
      case 'Convolution 2D':
        return `Conv2D:${layer.params.filters}:${layer.params.activation}`;
      case 'Max Pooling':
        return 'MaxPool';
      case 'Global Average Pooling':
        return 'GlobalAvgPool';
      case 'Flatten':
        return 'Flatten';
      case 'Dropout':
        return `Drop:${layer.params.rate}`;
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
      // Transform the architecture to the correct format
      const architectureToSave = entry.architecture.map(layer => ({
        type: layer.type,
        params: layer.params
      }));
      
      const architectureString = JSON.stringify(architectureToSave);
      
      // Save to localStorage
      localStorage.setItem('defaultArchitecture', architectureString);
      setSelectedDefault(architectureString);
      
      // Save to backend
      await setDefaultArchitecture(architectureToSave);
      
      if (onSetDefault) {
        onSetDefault(entry.architecture);
      }

      console.log('Successfully set default architecture:', architectureToSave);
    } catch (error) {
      console.error('Failed to set default architecture:', error);
      // You might want to show an error message to the user here
    }
  };

  // Sort history by test accuracy to easily identify the best model
  const sortedHistory = [...history].sort((a, b) => b.final_test_accuracy - a.final_test_accuracy);

  return (
    <Paper sx={{ p: 2, maxHeight: '80vh', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        Training History
      </Typography>
      {sortedHistory.map((entry, index) => {
        const isDefault = selectedDefault === JSON.stringify(entry.architecture);
        return (
          <Card key={entry.timestamp} sx={{ mb: 2, p: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="subtitle2" gutterBottom>
                Model #{history.length - index} - {new Date(entry.timestamp).toLocaleString()}
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
            
            <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
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
            </Typography>

            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 1
            }}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Train: {(entry.final_train_accuracy * 100).toFixed(2)}% | 
                  Test: {(entry.final_test_accuracy * 100).toFixed(2)}%
                </Typography>
                <Typography variant="caption" color="text.secondary">
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
        )
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