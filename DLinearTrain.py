import torch
import torch.optim as optim
import torch.nn as nn
from data_loaders.data_provider import data_provider
from args import Args
from models.DLinear import DLinear  

# Load arguments
args = Args()
args.model_type = 'DLinear'

# Limit dataset size for debugging/testing
data_limit = 1024

# Load data
train_set, train_loader = data_provider(args, flag='train', limit=data_limit)
test_set, test_loader = data_provider(args, flag='test', limit=data_limit)

# Initialize model
model = DLinear(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training loop
train_losses = []
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{args.epochs}:")

    for batch_idx, (batch_x, _, _, _, target) in enumerate(train_loader):
        # Move data to device
        batch_x, target = batch_x.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)

        # Loss calculation
        loss = criterion(outputs, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Average epoch loss
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Avg Loss: {avg_loss:.4f}")

# Plot training loss
import matplotlib.pyplot as plt
plt.plot(train_losses, marker='o')
plt.title('Training Loss - Decomposition Linear Model')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.show()
