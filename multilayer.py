import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# Load the downloaded Apple stock price data
data = pd.read_csv("apple.csv")

# Extract the relevant columns: 'Date', 'High', and 'Low'
data = data[['Date', 'High', 'Low']]

# Normalize the 'High' and 'Low' columns
scaler = MinMaxScaler(feature_range=(0, 1))
data[['High', 'Low']] = scaler.fit_transform(data[['High', 'Low']])

# Convert the 'Date' column to numerical values (Unix timestamps)
data['Date'] = pd.to_datetime(data['Date']).apply(lambda x: x.timestamp())

# Split the data into input features and target variable
X = data[['Date', 'High', 'Low']]
y = data['High']

# Convert the data to PyTorch tensors
X = torch.Tensor(X.values)
y = torch.Tensor(y.values).view(-1, 1)

# Define the MLP model with regularization
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Set the dimensions for input, hidden, and output layers
input_dim = X.shape[1]
hidden_dim = 64
output_dim = 1

# Define the hyperparameters
learning_rate = 0.001
weight_decay = 0.001
num_epochs = 100
num_folds = 5

# Initialize the K-Fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store the evaluation metrics for each fold
train_losses = []
val_losses = []

# Perform K-Fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold: {fold + 1}")
    
    # Split the data into training and validation sets for the current fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Create an instance of the MLP model
    model = MLP(input_dim, hidden_dim, output_dim)

    # Define the loss function and optimizer with weight decay
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the training loss for every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Evaluate the model on the training set
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train)
        train_loss = criterion(train_predictions, y_train)
        train_losses.append(train_loss.item())

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, y_val)
        val_losses.append(val_loss.item())

    # Print the evaluation metrics for the current fold
    print(f"Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}\n")

# Calculate the average evaluation metrics across all folds
avg_train_loss = sum(train_losses) / num_folds
avg_val_loss = sum(val_losses) / num_folds

# Print the average evaluation metrics
print(f"Avg Train Loss: {avg_train_loss}, Avg Validation Loss: {avg_val_loss}")
