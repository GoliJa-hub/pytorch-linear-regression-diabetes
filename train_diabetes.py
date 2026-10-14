import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # <--- 1. Import matplotlib

# 1. Load the real medical dataset from Scikit-Learn
print("Loading diabetes dataset...")
diabetes = load_diabetes()
X_raw, y_raw = diabetes.data, diabetes.target

# 2. Scale the data (helps gradient descent run smoothly and fast)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Convert Numpy arrays into PyTorch Tensors
# PyTorch requires floats for math operations (torch.float32)
X_train = torch.tensor(X_scaled, dtype=torch.float32)
# y_train needs an extra dimension (.unsqueeze(1)) to match shape [n_samples, 1]
y_train = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1)

# Automatically find how many features we have (should be 10 for this dataset)
n_features = X_train.shape[1]

# 4. Hyperparameters
learning_rate = 0.1
n_epochs = 100

# 5. Set up the Model, Optimizer, and Loss Function
torch.manual_seed(42)  # Ensures reproducible results
model = nn.Linear(in_features=n_features, out_features=1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 6. The Training Loop
# Create an empty list to store our loss values over time
loss_history = []
print("Starting training...")
for epoch in range(n_epochs):
    # Forward pass: compute predictions
    y_pred = model(X_train)
    
    # Compute the loss (error)
    loss = criterion(y_pred, y_train)
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Update weights automatically
    optimizer.step()
    
    # Clear gradients for the next epoch
    optimizer.zero_grad()
    
    #Save the loss value for this epoch
    loss_history.append(loss.item())
    
    # Print progress every 10 epochs so the screen isn't flooded
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

# 7. Plot the results after training finishes
plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='b', linewidth=2)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()

print("Training complete! Your model successfully learned from real medical data.")