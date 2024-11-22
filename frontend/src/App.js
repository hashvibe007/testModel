import React, { useState, useEffect } from 'react';
import { Container, Box, Grid } from '@mui/material';
import LayerPalette from './components/LayerPalette';
import NetworkBuilder from './components/NetworkBuilder';
import TrainingConfig from './components/TrainingConfig';
import TrainingResults from './components/TrainingResults';
import TrainingHistory from './components/TrainingHistory';
import { getTrainingHistory } from './services/api';
import './App.css';

function App() {
  const [layers, setLayers] = useState([]);
  const [trainingResults, setTrainingResults] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState([]);

  const loadTrainingHistory = async () => {
    try {
      const history = await getTrainingHistory();
      console.log('Loaded training history:', history);
      setTrainingHistory(history || []);
    } catch (error) {
      console.error('Failed to load training history:', error);
      setTrainingHistory([]);
    }
  };

  useEffect(() => {
    loadTrainingHistory();
  }, []);

  const layerTypes = {
    conv2d: {
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
    maxpool: {
      type: 'Max Pooling',
      icon: '🔲',
      defaultParams: {
        poolSize: 2,
        stride: 2
      }
    },
    globalavgpool: {
      type: 'Global Average Pooling',
      icon: '🌐',
      defaultParams: {}
    },
    flatten: {
      type: 'Flatten',
      icon: '📄',
      defaultParams: {}
    },
    dropout: {
      type: 'Dropout',
      icon: '💧',
      defaultParams: {
        rate: 0.25
      }
    },
    dense: {
      type: 'Fully Connected',
      icon: '🔌',
      defaultParams: {
        units: 128,
        activation: 'relu'
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const layerType = e.dataTransfer.getData('layerType');
    const layerTemplate = layerTypes[layerType];
    
    if (layerTemplate) {
      const newLayer = {
        id: `${layerType}-${Date.now()}`,
        type: layerTemplate.type,
        icon: layerTemplate.icon,
        defaultParams: { ...layerTemplate.defaultParams }
      };
      
      if (layerType === 'flatten' || layerType === 'globalavgpool') {
        if (layers.some(layer => 
          layer.type === 'Flatten' || layer.type === 'Global Average Pooling'
        )) {
          return;
        }
        const hasConvBefore = layers.some(layer => 
          layer.type === 'Convolution 2D' || layer.type === 'Max Pooling'
        );
        const hasDenseAfter = layers.some(layer => layer.type === 'Fully Connected');
        if (!hasConvBefore || hasDenseAfter) {
          return;
        }
      }
      
      setLayers([...layers, newLayer]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleLayerUpdate = (index, params) => {
    const newLayers = [...layers];
    newLayers[index] = { ...newLayers[index], defaultParams: params };
    setLayers(newLayers);
  };

  const handleLayerDelete = (index) => {
    const newLayers = [...layers];
    newLayers.splice(index, 1);
    setLayers(newLayers);
  };

  const getLayerTypeId = (layerType) => {
    switch (layerType) {
      case 'Convolution 2D':
        return 'conv2d';
      case 'Max Pooling':
        return 'maxpool';
      case 'Global Average Pooling':
        return 'globalavgpool';
      case 'Flatten':
        return 'flatten';
      case 'Dropout':
        return 'dropout';
      case 'Fully Connected':
        return 'dense';
      default:
        return layerType.toLowerCase().replace(' ', '');
    }
  };

  const handleSetDefaultArchitecture = (architecture) => {
    try {
      const uiArchitecture = architecture.map(layer => {
        const layerTypeId = getLayerTypeId(layer.type);
        const layerTemplate = layerTypes[layerTypeId];
        
        if (!layerTemplate) {
          console.error(`Unknown layer type: ${layer.type}`);
          return null;
        }

        return {
          id: `${layerTypeId}-${Date.now()}-${Math.random()}`,
          type: layer.type,
          icon: layerTemplate.icon,
          defaultParams: layer.params
        };
      }).filter(layer => layer !== null);

      setLayers(uiArchitecture);
      console.log('Set default architecture:', uiArchitecture);
    } catch (error) {
      console.error('Error setting default architecture:', error);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <h1>MNIST Training Dashboard</h1>
        <Grid container spacing={2}>
          <Grid item xs={12} md={8}>
            <LayerPalette />
            <Box
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              sx={{ 
                minHeight: '200px',
                border: '2px dashed #ccc',
                borderRadius: '4px',
                mb: 2,
                p: 2
              }}
            >
              <NetworkBuilder
                layers={layers}
                onLayerUpdate={handleLayerUpdate}
                onLayerDelete={handleLayerDelete}
              />
            </Box>
            <TrainingConfig 
              networkLayers={layers}
              setTrainingResults={setTrainingResults}
              isTraining={isTraining}
              setIsTraining={setIsTraining}
              onTrainingComplete={loadTrainingHistory}
            />
            <TrainingResults results={trainingResults} />
          </Grid>
          <Grid item xs={12} md={4}>
            <TrainingHistory 
              history={trainingHistory} 
              onSetDefault={handleSetDefaultArchitecture}
            />
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}

export default App; 