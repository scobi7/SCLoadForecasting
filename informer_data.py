import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class InformerDataset(Dataset):
    def __init__(self, data, look_back=96, pred_len=24, features='S', target='dayton_mw', freq='h'):
        self.look_back = look_back
        self.pred_len = pred_len
        self.data = data
        self.features = features
        self.target = target
        self.freq = freq
        
        self.scaler = StandardScaler()

        # Ensure datetime column is parsed and set as the index
        if 'datetime' not in data.columns:
            raise ValueError("The DataFrame must contain a 'datetime' column.")
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)

        self.data = data.copy()
        self.data[['temperature', 'precipitation', 'humidity', 'wind_speed', self.target]] = (
            self.scaler.fit_transform(self.data[['temperature', 'precipitation', 'humidity', 'wind_speed', self.target]])
        )

    def __len__(self):
        return len(self.data) - self.look_back - self.pred_len

    def __getitem__(self, idx):
        # Encoder input
        x_enc = self.data.iloc[idx: idx + self.look_back][['temperature', 'precipitation', 'humidity', 'wind_speed']].values
        
        # Decoder input
        x_dec = self.data.iloc[idx + self.look_back - self.pred_len: idx + self.look_back][['temperature', 'precipitation', 'humidity', 'wind_speed']].values
        
        # Target
        target = self.data.iloc[idx + self.look_back: idx + self.look_back + self.pred_len][self.target].values

        # Time encodings for encoder and decoder
        x_mark_enc = self.data.iloc[idx: idx + self.look_back].index.map(self._encode_time).tolist()
        x_mark_dec = self.data.iloc[idx + self.look_back - self.pred_len: idx + self.look_back].index.map(self._encode_time).tolist()

        # Convert to tensors
        x_enc = torch.tensor(x_enc, dtype=torch.float32)
        x_dec = torch.tensor(x_dec, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        x_mark_enc = torch.tensor(np.stack(x_mark_enc), dtype=torch.float32)
        x_mark_dec = torch.tensor(np.stack(x_mark_dec), dtype=torch.float32)

        return x_enc, x_dec, x_mark_enc, x_mark_dec, target

    def _encode_time(self, datetime):
        """
        Encodes datetime into numerical features (month, day, weekday, hour).
        """
        return [
            datetime.month,
            datetime.day,
            datetime.weekday(),
            datetime.hour
        ]
