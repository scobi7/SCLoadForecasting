import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from args import Args
from data_provider import data_provider
from model import InformerModel

# Load arguments
args = Args()
args.model_type = 'Informer'

# Limit data size for testing
data_limit = 512  # Adjust this value as needed for testing

# Load data
train_set, train_loader = data_provider(args, flag='train', limit=data_limit)
test_set, test_loader = data_provider(args, flag='test', limit=data_limit)

# Initialize Informer model
model = InformerModel(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training loop
train_losses = []
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{args.epochs}:")

    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark, target) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x_mark, batch_y_mark = batch_x_mark.to(device), batch_y_mark.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x, batch_x_mark, batch_y[:, :args.label_len, :], batch_y_mark[:, :args.label_len, :])
        #print(f"Batch {batch_idx+1}/{len(train_loader)}") #: outputs.shape = {outputs.shape}

        # Calculate loss
        loss = criterion(outputs.squeeze(), target.squeeze())
        #print(f"Batch {batch_idx+1}/{len(train_loader)}: loss = {loss.item()}")

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Avg Loss: {avg_loss:.4f}")
    #print(f"Epoch [{epoch+1}/{args.epochs}], Avg Loss: {avg_loss:.4f}")

# Plot training loss
plt.plot(train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.show()