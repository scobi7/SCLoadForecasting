import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as mp
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class LoadForecastingSNN(Dataset):
    def __init__(self, data, look_back=5):
        self.data = data
        self.look_back = look_back

        # Normalize the data (optional but often helpful in time series forecasting)
        self.data[['temperature', 'precipitation', 'humidity', 'wind_speed', 'dayton_mw']] = (
            self.data[['temperature', 'precipitation', 'humidity', 'wind_speed', 'dayton_mw']].apply(
                lambda x: (x - x.mean()) / x.std())
        )

    def __len__(self):
        return len(self.data) - self.look_back - 1

    def __getitem__(self, idx):
        # data
        x = self.data.iloc[idx: idx + self.look_back][['temperature', 'precipitation', 'humidity', 'wind_speed']].values
        # target data
        y = self.data.iloc[idx + self.look_back + 1]['dayton_mw']
        
        # Convert to tensors (or does this have to be vectors)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y
    

def dataloader(data_path, batch_size=32, look_back=5, test_size=0.25):
    # Load data
    data = pd.read_csv(data_path, parse_dates=['datetime'])
    
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)  
    
    # Initialize training and testing datasets
    train_dataset = LoadForecastingSNN(train_data, look_back=look_back)
    test_dataset = LoadForecastingSNN(test_data, look_back=look_back)
    
    # Create DataLoaders for each
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
    
    return train_loader, test_loader

# Usage
data_path = '/Users/scobi/Desktop/SCLoadForecasting/combinedDaytonData_fill.csv'
batch_size = 32
look_back = 5

train_loader, test_loader = dataloader(data_path, batch_size=batch_size, look_back=look_back, test_size=0.25)

# Verify split by checking the lengths of each DataLoader
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Testing samples: {len(test_loader.dataset)}")
    

