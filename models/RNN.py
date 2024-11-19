import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def create_time_series_dataset(data, lookback_window, forecasting_horizon=1, test_size=0.25):
    x_processed = []
    y_processed = []

    # Create sliding windows
    for i in range(len(data) - lookback_window - forecasting_horizon + 1):
        x_window = data[i:i + lookback_window, :-1]  # Use all features except target
        y_value = data[i + lookback_window + forecasting_horizon - 1, -1]  # Target is the load forecast
        x_processed.append(x_window)
        y_processed.append(y_value)

    # Convert to numpy arrays
    x_processed = np.array(x_processed)
    y_processed = np.array(y_processed)

    # Print shapes and contents of x_processed and y_processed
    print("Shape of x_processed (input windows):", x_processed.shape)  # Expected: (num_samples, lookback_window, num_features)
    print("Sample x_processed (first input window):", x_processed[0])  # Prints first input window
    print("Shape of y_processed (targets):", y_processed.shape)        # Expected: (num_samples,)
    print("Sample y_processed (first target):", y_processed[0])        # Prints first target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x_processed, y_processed, test_size=test_size, shuffle=False)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    return train_loader, test_loader

class SimpleRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, output_size=1, num_layers=5):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #rnn layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        #output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN forward pass
        out, _ = self.rnn(x, h0)
        
        # Apply ReLU activation to the last time step's output
        out = F.relu(out[:, -1, :])
        
        # Fully connected layer
        out = self.fc(out)
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=1, num_layers=5, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer with Dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_size)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM Forward Pass
        out, _ = self.lstm(x, (h0, c0))

        # Apply Batch Normalization and Activation to Last Time Step's Output
        out = F.leaky_relu(self.bn(out[:, -1, :]))

        # Fully Connected Layer
        out = self.fc(out)
        return out
    
class SimpleGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, output_size=1, num_layers=5, dropout=0.2):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU Layer with Dropout
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_size)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU Forward Pass
        out, _ = self.gru(x, h0)

        # Apply Batch Normalization and Activation to Last Time Step's Output
        out = F.leaky_relu(self.bn(out[:, -1, :]))

        # Fully Connected Layer
        out = self.fc(out)
        return out