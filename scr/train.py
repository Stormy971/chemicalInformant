import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import ChemNet
from data_utils import load_dataset

def train(csv_path="data/delaney.csv", epochs=500, batch_size=32, lr=1e-3):
    # Step 1: Load dataset
    X, y = load_dataset(csv_path, smiles_col="SMILES", target_col="measured log(solubility:mol/L)")

    # Step 2: Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Step 3: Create DataLoader for batching
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 4: Initialize model, loss, optimizer
    model = ChemNet(input_dim=2048, hidden1_dim=1024, hidden2_dim=512, output_dim=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Step 5: Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()           # Clear gradients
            pred = model(xb)               # Forward pass
            loss = criterion(pred, yb)     # Compute loss
            loss.backward()                # Backpropagate
            optimizer.step()               # Update weights
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), "models/chemnet_weights.pth")
    print("Model weights saved to models/chemnet_weights.pth")

if __name__ == "__main__":
    train()
