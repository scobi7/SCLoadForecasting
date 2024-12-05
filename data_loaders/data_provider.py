import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
from LTSF.utils.timefeatures import time_features
import numpy as np

class DataProvider(Dataset):
    def __init__(self, args, flag='train'):
        self.enc_in = args.enc_in
        self.dec_in = args.dec_in
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.features = args.features
        self.target = args.target
        self.scale = args.scale
        self.timeenc = args.timeenc
        self.freq = args.freq
        self.train_only = args.train_only
        self.batch_size = args.batch_size

        self.root_path = args.root_path
        self.data_path = args.data_path
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        """
        Reads and preprocesses the dataset based on the arguments.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        self.target_column = df_raw.columns.get_loc(self.target)

        # Ensure the target column exists
        if self.target not in df_raw.columns:
            raise ValueError(f"Target column '{self.target}' not found in the dataset.")

        # Separate data columns based on features
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('datetime')

        # Split into train, validation, and test sets
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self._get_type_index()]
        border2 = border2s[self._get_type_index()]

        # Prepare data based on features
        if self.features in ['M', 'MS']:
            df_data = df_raw[cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Prepare timestamps
        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp['datetime'])

        # Encode time features
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['datetime'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['datetime'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['datetime'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['datetime'].apply(lambda row: row.hour)
            self.data_stamp = df_stamp.drop(['datetime'], axis=1).values
        elif self.timeenc == 1:
            self.data_stamp = time_features(pd.to_datetime(df_stamp['datetime'].values), freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        """
        Returns a data sample for the given index.
        """
        x_enc = self.data_x[idx: idx + self.seq_len]
        x_dec = self.data_x[idx + self.seq_len - self.label_len: idx + self.seq_len]
        target = self.data_y[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        x_enc.shape == (self.batch_size, self.seq_len, self.enc_in)  # Expected: (16, 96, 5)
        x_dec.shape == (self.batch_size, self.label_len, self.dec_in)  # Expected: (16, 48, 5)

        x_mark_enc = self.data_stamp[idx: idx + self.seq_len]
        x_mark_dec = self.data_stamp[idx + self.seq_len - self.label_len: idx + self.seq_len]

        # Convert to PyTorch tensors
        x_enc = torch.tensor(x_enc, dtype=torch.float32)
        x_dec = torch.tensor(x_dec, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32)
        x_mark_dec = torch.tensor(x_mark_dec, dtype=torch.float32)

        return x_enc, x_dec, x_mark_enc, x_mark_dec, target


    def inverse_transform(self, data):
        """
        Reverses scaling transformation for predictions.
        """
        return self.scaler.inverse_transform(data)

    def _get_type_index(self):
        type_map = {'train': 0, 'val': 1, 'test': 2}
        return type_map[self.flag]


def data_provider(args, flag, limit=None):
    dataset = DataProvider(args, flag)

    # Apply optional size limit
    if limit:
        dataset.data_x = dataset.data_x[:limit]
        dataset.data_y = dataset.data_y[:limit]
        dataset.data_stamp = dataset.data_stamp[:limit]

    # Define DataLoader parameters
    shuffle_flag = flag == 'train'
    drop_last = flag == 'train'

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    print(f"Dataset Size: {len(dataset)}, Batch Size: {args.batch_size}, Number of Batches: {len(data_loader)}")
    return dataset, data_loader
