import React, { useState } from 'react';
import { Paper, Typography, Box, IconButton, Tooltip, Collapse } from '@mui/material';
import { DragHandle, KeyboardArrowLeft, KeyboardArrowRight } from '@mui/icons-material';
import LayerCard from './LayerCard';

const LayerPalette = ({ layerTypes }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 20, y: 100 }); // Initial position

  const handleDragStart = (e) => {
    if (e.target.classList.contains('drag-handle')) {
      setIsDragging(true);
      // Store the initial mouse offset from the palette's top-left corner
      const rect = e.currentTarget.getBoundingClientRect();
      e.dataTransfer.setData('offset', JSON.stringify({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      }));
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (isDragging) {
      const offset = JSON.parse(e.dataTransfer.getData('offset'));
      setPosition({
        x: e.clientX - offset.x,
        y: e.clientY - offset.y
      });
      setIsDragging(false);
    }
  };

  return (
    <Paper
      sx={{
        position: 'fixed',
        top: position.y,
        left: position.x,
        zIndex: 1000,
        width: isCollapsed ? 'auto' : 200,
        transition: 'width 0.3s',
        cursor: isDragging ? 'grabbing' : 'auto',
        boxShadow: 3
      }}
      draggable
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <Box sx={{ 
        p: 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: 'primary.main',
        color: 'white',
        cursor: 'grab'
      }} 
      className="drag-handle"
      >
        {!isCollapsed && <Typography variant="subtitle1">Available Layers</Typography>}
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <DragHandle sx={{ cursor: 'grab' }} />
          <IconButton 
            size="small" 
            onClick={() => setIsCollapsed(!isCollapsed)}
            sx={{ color: 'white', ml: 1 }}
          >
            {isCollapsed ? <KeyboardArrowRight /> : <KeyboardArrowLeft />}
          </IconButton>
        </Box>
      </Box>

      <Collapse in={!isCollapsed}>
        <Box sx={{ 
          p: 2,
          display: 'flex', 
          flexDirection: 'column',
          gap: 1,
          maxHeight: '70vh',
          overflowY: 'auto'
        }}>
          {Object.entries(layerTypes).map(([id, layer]) => (
            <Box
              key={id}
              sx={{
                cursor: 'grab',
                '&:active': { cursor: 'grabbing' }
              }}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData('layerType', id);
              }}
            >
              <LayerCard layer={{ ...layer, id }} isTemplate={true} />
            </Box>
          ))}
        </Box>
      </Collapse>
    </Paper>
  );
};

export default LayerPalette; 