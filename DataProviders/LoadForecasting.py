import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class LoadForecastingCustomDataset(Dataset):
    def __init__(self, root_path=None, data_path=None, flag='train', size=None,
                 features='S', target='dayton_mw', timeenc=0, freq='h', train_only=False, look_back=5):
        # Initialize parameters
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.size = size
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.look_back = look_back

        # Load and normalize data
        self.data = self.load_data()
        self.scaler = StandardScaler()
        self.data[['temperature', 'precipitation', 'humidity', 'wind_speed', 'dayton_mw']] = (
            self.scaler.fit_transform(self.data[['temperature', 'precipitation', 'humidity', 'wind_speed', 'dayton_mw']])
        )

    def load_data(self):
        # Load data from the specified path
        if self.data_path is not None:
            data = pd.read_csv(self.data_path, parse_dates=['datetime'])
            data = data[['temperature', 'precipitation', 'humidity', 'wind_speed', 'dayton_mw']]
        else:
            raise ValueError("Data path must be provided to load the dataset.")
        return data

    def __len__(self):
        return len(self.data) - self.look_back - 1

    def __getitem__(self, idx):
        # Get input features (temperature, precipitation, humidity, wind_speed) for `look_back` time steps
        x = self.data.iloc[idx: idx + self.look_back][['temperature', 'precipitation', 'humidity', 'wind_speed']].values
        # Get the target (dayton_mw) for the prediction time step
        y = self.data.iloc[idx + self.look_back + 1]['dayton_mw']
        
        # Convert data to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def inverse_transform(self, data):
        # Inverse transformation for scaled data
        return self.scaler.inverse_transform(data)
