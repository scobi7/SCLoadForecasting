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
    

