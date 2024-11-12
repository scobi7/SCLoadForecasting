import torch
import torch.optim as optim
import torch.nn as nn 
from torch.utils.data import DataLoader
from model import SimpleRNN, create_time_series_dataset
from LoadForecasting import LoadForecastingSNN
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt

# Load data
data_path = '/Users/scobi/Desktop/SCLoadForecasting/combinedDaytonData_fill.csv'
df = pd.read_csv(data_path, parse_dates=['datetime'])
df = df[['temperature', 'precipitation', 'humidity', 'wind_speed', 'dayton_mw']]

# Standardize features and target
feature_means = df[['temperature', 'precipitation', 'humidity', 'wind_speed']].mean()
feature_stds = df[['temperature', 'precipitation', 'humidity', 'wind_speed']].std()

target_mean = df['dayton_mw'].mean()
target_std = df['dayton_mw'].std()

# Standardize the features and target
df[['temperature', 'precipitation', 'humidity', 'wind_speed']] = (df[['temperature', 'precipitation', 'humidity', 'wind_speed']] - feature_means) / feature_stds
df['dayton_mw'] = (df['dayton_mw'] - target_mean) / target_std

data = df.values  # Convert to numpy array for processing


# Parameters
lookback_window = 5  # Number of time steps to look back
forecasting_horizon = 1  # Number of time steps to predict ahead
batch_size = 1
num_epochs = 10
learning_rate = 0.001

# Normalize the data and create train and test loaders
train_loader, test_loader = create_time_series_dataset(data, lookback_window, forecasting_horizon)

# Initialize model, loss function, and optimizer
model = SimpleRNN(input_size=4, hidden_size=128, output_size=1, num_layers=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
train_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)
        
        # Forward pass
        output = model(inputs)
        loss = criterion(output, target.unsqueeze(1))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Plot training loss
plt.plot(train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.show()

