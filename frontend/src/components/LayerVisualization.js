import React, { useState, useEffect } from 'react';
import { Box, Typography, IconButton, Popover, Grid, CircularProgress } from '@mui/material';
import { Visibility } from '@mui/icons-material';

const LayerVisualization = ({ layer, dimensions, previousLayers }) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [visualizations, setVisualizations] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleClick = async (event) => {
    setAnchorEl(event.currentTarget);
    if (!visualizations.length) {
      await loadVisualizations();
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const loadVisualizations = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/layer-visualization', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          layer: {
            type: layer.type,
            params: layer.defaultParams
          },
          previous_layers: previousLayers.map(l => ({
            type: l.type,
            params: l.defaultParams
          }))
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch visualizations');
      }

      const data = await response.json();
      setVisualizations(data.visualizations);
    } catch (error) {
      console.error('Error loading visualizations:', error);
    } finally {
      setLoading(false);
    }
  };

  const open = Boolean(anchorEl);

  return (
    <>
      <IconButton 
        size="small" 
        onClick={handleClick}
        sx={{ ml: 1 }}
        title="View feature maps"
      >
        <Visibility fontSize="small" />
      </IconButton>
      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'center',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'center',
          horizontal: 'left',
        }}
      >
        <Box sx={{ p: 2, maxWidth: 400 }}>
          <Typography variant="subtitle2" gutterBottom>
            Feature Visualization - {layer.type}
          </Typography>
          <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
            Output shape: {dimensions?.output.join(' × ')}
          </Typography>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress size={24} />
            </Box>
          ) : (
            <Grid container spacing={1} sx={{ mt: 1 }}>
              {visualizations.map((imgStr, idx) => (
                <Grid item xs={4} key={idx}>
                  <Box
                    component="img"
                    src={`data:image/png;base64,${imgStr}`}
                    alt={`Feature map ${idx + 1}`}
                    sx={{
                      width: '100%',
                      height: 'auto',
                      display: 'block',
                      border: '1px solid #eee',
                      borderRadius: 1
                    }}
                  />
                </Grid>
              ))}
              {!visualizations.length && !loading && (
                <Grid item xs={12}>
                  <Box 
                    sx={{ 
                      width: '100%', 
                      height: 200, 
                      backgroundColor: '#f5f5f5',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    <Typography variant="caption" color="text.secondary">
                      No feature maps available for this layer type
                    </Typography>
                  </Box>
                </Grid>
              )}
            </Grid>
          )}
        </Box>
      </Popover>
    </>
  );
};

export default LayerVisualization; 