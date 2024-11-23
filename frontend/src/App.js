import React, { useState, useEffect } from 'react';
import { Container, Box, Grid, Paper } from '@mui/material';
import LayerPalette from './components/LayerPalette';
import NetworkBuilder from './components/NetworkBuilder';
import TrainingConfig from './components/TrainingConfig';
import TrainingResults from './components/TrainingResults';
import TrainingHistory from './components/TrainingHistory';
import DataSourceSelector from './components/DataSourceSelector';
import { getTrainingHistory } from './services/api';
import './App.css';

function App() {
  const [layers, setLayers] = useState([]);
  const [trainingResults, setTrainingResults] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [dataSource, setDataSource] = useState('mnist');

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
    conv1x1: {
      type: 'Convolution 1x1',
      icon: '🔍',
      defaultParams: {
        filters: 32,
        activation: 'relu'
      }
    },
    batchnorm: {
      type: 'Batch Normalization',
      icon: '📊',
      defaultParams: {
        momentum: 0.99,
        epsilon: 0.001
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
        units: 10,
        activation: 'softmax'
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
      
      // Validate layer placement
      const newLayers = [...layers];
      let canAdd = true;
      
      // Check if trying to add Flatten or Global Average Pooling
      if (layerType === 'flatten' || layerType === 'globalavgpool') {
        // Check if one already exists
        if (layers.some(layer => 
          layer.type === 'Flatten' || layer.type === 'Global Average Pooling'
        )) {
          canAdd = false;
        }
        
        // Check if there are conv layers before and no dense layers after
        const hasConvBefore = layers.some(layer => 
          layer.type === 'Convolution 2D' || 
          layer.type === 'Convolution 1x1' || 
          layer.type === 'Max Pooling'
        );
        const hasDenseAfter = layers.some(layer => layer.type === 'Fully Connected');
        
        if (!hasConvBefore || hasDenseAfter) {
          canAdd = false;
        }
      }
      
      // Check if trying to add conv/pool after dense
      if ((layerType === 'conv2d' || layerType === 'conv1x1' || layerType === 'maxpool') && 
          layers.some(layer => layer.type === 'Fully Connected')) {
        canAdd = false;
      }
      
      // Add layer if validation passes
      if (canAdd) {
        setLayers([...layers, newLayer]);
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleLayerUpdate = (indexOrNewLayers, params) => {
    if (Array.isArray(indexOrNewLayers)) {
      // Handle reordering
      setLayers(indexOrNewLayers);
    } else {
      // Handle parameter updates
      const newLayers = [...layers];
      newLayers[indexOrNewLayers] = { ...newLayers[indexOrNewLayers], defaultParams: params };
      setLayers(newLayers);
    }
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
      case 'Convolution 1x1':
        return 'conv1x1';
      case 'Batch Normalization':
        return 'batchnorm';
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

  const handleDataSourceChange = (newSource) => {
    setDataSource(newSource);
    setLayers([]);
    setTrainingResults([]);
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 4 }}>
        <h1>Neural Network Training Dashboard</h1>
        <Grid container spacing={2}>
          {/* Left Sidebar */}
          <Grid item xs={12} md={3}>
            <Box sx={{ position: 'sticky', top: 20 }}>
              <TrainingConfig 
                networkLayers={layers}
                setTrainingResults={setTrainingResults}
                isTraining={isTraining}
                setIsTraining={setIsTraining}
                onTrainingComplete={loadTrainingHistory}
                dataSource={dataSource}
              />
              <DataSourceSelector 
                selectedSource={dataSource}
                onSourceChange={handleDataSourceChange}
              />
            </Box>
          </Grid>

          {/* Main Content Area */}
          <Grid item xs={12} md={6}>
            <Box
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              sx={{ 
                minHeight: '200px',
                backgroundColor: '#f8f8f8',
                borderRadius: '4px',
                mb: 2,
                p: 2
              }}
            >
              <NetworkBuilder
                layers={layers}
                onLayerUpdate={handleLayerUpdate}
                onLayerDelete={handleLayerDelete}
                dataSource={dataSource}
                layerTypes={layerTypes}
              />
            </Box>
            <TrainingResults results={trainingResults} />
          </Grid>

          {/* Right Sidebar */}
          <Grid item xs={12} md={3}>
            <Box sx={{ position: 'sticky', top: 20 }}>
              <TrainingHistory 
                history={trainingHistory} 
                onSetDefault={handleSetDefaultArchitecture}
              />
            </Box>
          </Grid>
        </Grid>
      </Box>
      
      {/* Floating Layer Palette */}
      <LayerPalette layerTypes={layerTypes} />
    </Container>
  );
}

export default App; 