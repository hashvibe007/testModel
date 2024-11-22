import React from 'react';
import {
  Paper,
  Typography,
  FormControlLabel,
  Switch,
  Slider,
  Box,
  Collapse,
  Grid
} from '@mui/material';
import ImageAugmentationPreview from './ImageAugmentationPreview';

const ImageAugmentationConfig = ({ enabled, config, onChange }) => {
  const handleChange = (key, value) => {
    onChange({
      ...config,
      [key]: value
    });
  };

  return (
    <>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Image Augmentation</Typography>
          <FormControlLabel
            control={
              <Switch
                checked={enabled}
                onChange={(e) => onChange({ ...config, enabled: e.target.checked })}
              />
            }
            label="Enable"
          />
        </Box>

        <Collapse in={enabled}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography gutterBottom>Random Rotation (degrees)</Typography>
              <Slider
                value={config.rotation}
                onChange={(_, value) => handleChange('rotation', value)}
                min={0}
                max={30}
                valueLabelDisplay="auto"
                disabled={!enabled}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>Random Zoom</Typography>
              <Slider
                value={config.zoom}
                onChange={(_, value) => handleChange('zoom', value)}
                min={0}
                max={0.2}
                step={0.01}
                valueLabelDisplay="auto"
                disabled={!enabled}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>Width Shift</Typography>
              <Slider
                value={config.width_shift}
                onChange={(_, value) => handleChange('width_shift', value)}
                min={0}
                max={0.2}
                step={0.01}
                valueLabelDisplay="auto"
                disabled={!enabled}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>Height Shift</Typography>
              <Slider
                value={config.height_shift}
                onChange={(_, value) => handleChange('height_shift', value)}
                min={0}
                max={0.2}
                step={0.01}
                valueLabelDisplay="auto"
                disabled={!enabled}
              />
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.horizontal_flip}
                    onChange={(e) => handleChange('horizontal_flip', e.target.checked)}
                    disabled={!enabled}
                  />
                }
                label="Horizontal Flip"
              />
            </Grid>
          </Grid>
        </Collapse>
      </Paper>
      <ImageAugmentationPreview config={config} />
    </>
  );
};

export default ImageAugmentationConfig; 