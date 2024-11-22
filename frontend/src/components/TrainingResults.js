import React from 'react';
import { Paper, Box } from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

const TrainingResults = ({ results }) => {
  if (results.length === 0) return null;

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ mb: 4 }}>
        <h3>Accuracy</h3>
        <LineChart width={600} height={300} data={results}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="train_accuracy"
            stroke="#8884d8"
            name="Training Accuracy"
          />
          <Line
            type="monotone"
            dataKey="test_accuracy"
            stroke="#82ca9d"
            name="Test Accuracy"
          />
        </LineChart>
      </Box>

      <Box>
        <h3>Loss</h3>
        <LineChart width={600} height={300} data={results}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="train_loss"
            stroke="#ff7300"
            name="Training Loss"
          />
          <Line
            type="monotone"
            dataKey="test_loss"
            stroke="#ff0000"
            name="Test Loss"
          />
        </LineChart>
      </Box>
    </Paper>
  );
};

export default TrainingResults; 