import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from DataProviders.data_provider import data_provider
from DataProviders.args import Args
from models.Informer import InformerModel  # Adjust this based on your file structure

# Initialize arguments
args = Args()

# Load train and test data
train_set, train_loader = data_provider(args, flag='train')
test_set, test_loader = data_provider(args, flag='test')

# Initialize Informer model
model = InformerModel(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses = []
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y, batch_x_mark, batch_y_mark, target in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x_mark, batch_y_mark = batch_x_mark.to(device), batch_y_mark.to(device)
        target = target.to(device)

        # Forward pass
        outputs = model(batch_x, batch_x_mark, batch_y[:, :args.label_len, :], batch_y_mark[:, :args.label_len, :])

        # Calculate loss
        loss = criterion(outputs.squeeze(), target.squeeze())


        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

# Plot training loss
plt.plot(train_losses, marker="o")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error Loss")
plt.show()
