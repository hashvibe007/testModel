import React, { useState, useEffect } from 'react';
import { 
  Paper, 
  Typography, 
  Grid, 
  Box,
  CircularProgress,
  Button
} from '@mui/material';
import { Refresh } from '@mui/icons-material';
import { getAugmentedImages } from '../services/api';

const ImageAugmentationPreview = ({ config }) => {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadAugmentedImages = async () => {
    if (!config.enabled) {
      setImages([]);
      return;
    }

    setLoading(true);
    try {
      const augmentedImages = await getAugmentedImages(config);
      setImages(augmentedImages);
    } catch (error) {
      console.error('Failed to load augmented images:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAugmentedImages();
  }, [config]);

  if (!config.enabled) {
    return null;
  }

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Augmentation Preview</Typography>
        <Button
          startIcon={<Refresh />}
          onClick={loadAugmentedImages}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={1}>
          {images.map((image, index) => (
            <Grid item xs={4} key={index}>
              <Box
                component="img"
                src={`data:image/png;base64,${image}`}
                alt={`Augmented sample ${index + 1}`}
                sx={{
                  width: '100%',
                  height: 'auto',
                  border: '1px solid #eee',
                  borderRadius: 1
                }}
              />
            </Grid>
          ))}
        </Grid>
      )}
    </Paper>
  );
};

export default ImageAugmentationPreview; 