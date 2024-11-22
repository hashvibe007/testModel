const API_BASE_URL = 'http://localhost:8000';

export const trainModel = async (config) => {
  try {
    console.log('Sending training config:', config);
    const response = await fetch(`${API_BASE_URL}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Training failed');
    }
    
    const data = await response.json();
    console.log('Training response:', data);
    return {
      results: data.results || [],
      history: data.history || []
    };
  } catch (error) {
    console.error('Training error:', error);
    throw error;
  }
};

export const getModelParams = async (convLayers) => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/model-params?conv_layers=${convLayers.join(',')}`
    );
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get model parameters');
    }
    
    return response.json();
  } catch (error) {
    console.error('Error fetching model parameters:', error);
    throw error;
  }
};

export const getTrainingHistory = async () => {
  try {
    console.log('Fetching training history...');
    const response = await fetch(`${API_BASE_URL}/history`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get training history');
    }
    
    const data = await response.json();
    console.log('Received training history:', data);
    return Array.isArray(data) ? data : [];
  } catch (error) {
    console.error('Error fetching training history:', error);
    return [];
  }
};

export const setDefaultArchitecture = async (config) => {
  try {
    console.log('Sending configuration to backend:', config);

    const response = await fetch(`${API_BASE_URL}/set-default-architecture`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to set default architecture');
    }
    
    return response.json();
  } catch (error) {
    console.error('Error setting default architecture:', error);
    throw error;
  }
};

export const getAugmentedImages = async (config) => {
  try {
    const response = await fetch(`${API_BASE_URL}/augmented-samples`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error('Failed to get augmented images');
    }

    const data = await response.json();
    return data.images;
  } catch (error) {
    console.error('Error getting augmented images:', error);
    throw error;
  }
}; 