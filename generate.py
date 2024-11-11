
import torch
import numpy as np
from torch.utils.data import DataLoader
from NeuMF import NeuMF, RatingDataset, matrix_to_triplets, train, predict_full_matrix, evaluate_from_matrix
import argparse
import torch.optim as optim
import torch.nn as nn
from RoundNet import train_with_early_stopping

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()

    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
#######################################
# Configuration
config = {
        'num_users': 610,  # Adjust based on your dataset
        'num_items': 4980,  # Adjust based on your dataset
        'latent_dim_mf': 8,
        'latent_dim_mlp': 32,
        'layers': [64, 32, 16, 8],  # MLP layer sizes
        'dropout_rate_mf': 0.2,
        'dropout_rate_mlp': 0.2,
        'lr': 0.001,
        'batch_size': 512,
        'epochs': 100,
    }


    # Convert to (user, item, rating) triplet format
train_triplets = matrix_to_triplets(table)
test_triplets = matrix_to_triplets(table)

    # Split validation data from training data (80% train, 20% val)
split_idx = int(0.8 * len(train_triplets))
val_triplets = train_triplets[split_idx:]
train_triplets = train_triplets[:split_idx]

    # Create Dataset and DataLoader objects
train_dataset = RatingDataset(train_triplets)
val_dataset = RatingDataset(val_triplets)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuMF(config).to(device)
criterion = nn.MSELoss()  # Use MSELoss for predicting continuous ratings
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train model with early stopping and loss tracking
train_losses, val_losses = train_with_early_stopping(model, criterion, optimizer, train_loader, val_loader, config, val_triplets, device, patience=8)

# Predict the entire matrix
full_matrix = predict_full_matrix(model, config['num_users'], config['num_items'], device)
print("Completed matrix prediction.")

    # Evaluate from the predicted matrix
rmse, accuracy = evaluate_from_matrix(full_matrix, train_triplets)
print(f"Test RMSE: {rmse:.4f}, Test Accuracy: {accuracy:.4f}")

print(accuracy)
table = full_matrix
####################################################
    # Save the completed table 
np.save("output.npy", table) ## DO NOT CHANGE THIS LINE
