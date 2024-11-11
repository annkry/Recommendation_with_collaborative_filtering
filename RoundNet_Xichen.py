import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

# Sigmoid-based approximation for the staircase function using PyTorch
def sigmoid_approx_torch(x, steps=5, k=10):
    approximation = torch.zeros_like(x)  # Initialize with zeros
    for n in range(steps):
        x_n = 0.5 + n
        A_n = 1
        approximation += A_n * torch.sigmoid(k * (x - x_n))
    return approximation

class NeuMF(nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()

        # MF part
        self.embedding_user_mf = nn.Embedding(config['num_users'], config['latent_dim_mf'])
        self.embedding_item_mf = nn.Embedding(config['num_items'], config['latent_dim_mf'])

        # MLP part
        self.embedding_user_mlp = nn.Embedding(config['num_users'], config['latent_dim_mlp'])
        self.embedding_item_mlp = nn.Embedding(config['num_items'], config['latent_dim_mlp'])

        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(config['layers'][:-1], config['layers'][1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        # Final logits layer
        self.logits = nn.Linear(config['layers'][-1] + config['latent_dim_mf'], 1)
        self.config = config

    def forward(self, user_indices, item_indices):
        # MLP embeddings
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        # MF embeddings
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # MF part
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        mf_vector = nn.Dropout(self.config['dropout_rate_mf'])(mf_vector)

        # MLP part
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        for fc in self.fc_layers:
            mlp_vector = fc(mlp_vector)
            mlp_vector = nn.ReLU()(mlp_vector)
        mlp_vector = nn.Dropout(self.config['dropout_rate_mlp'])(mlp_vector)

        # Concatenate MF and MLP parts
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.logits(vector)
        
        # Scale the logits to [0, 5] before applying the sigmoid approximation
        scaled_logits = 2.5 * (torch.tanh(logits) + 1)

        # Apply sigmoid-based approximation
        rounded_output = sigmoid_approx_torch(scaled_logits, steps=5, k=10)
        return rounded_output
    
# Custom dataset for triplet data (user, item, rating)
class RatingDataset(Dataset):
    def __init__(self, triplet_data):
        self.triplet_data = triplet_data

    def __len__(self):
        return len(self.triplet_data)

    def __getitem__(self, idx):
        user = self.triplet_data[idx, 0]
        item = self.triplet_data[idx, 1]
        rating = self.triplet_data[idx, 2]
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(rating, dtype=torch.float)

# Function to convert matrix data into (user_id, item_id, rating) triplet format
def matrix_to_triplets(matrix):
    triplets = []
    num_users, num_items = matrix.shape
    for user_id in range(num_users):
        for item_id in range(num_items):
            rating = matrix[user_id, item_id]
            if not np.isnan(rating):
                triplets.append([user_id, item_id, rating])
    return np.array(triplets)

# Training function with loss tracking
def train_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for users, items, ratings in train_loader:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)

        optimizer.zero_grad()
        outputs = model(users, items).squeeze()

        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Validation function to calculate validation loss
def evaluate_epoch(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for users, items, ratings in val_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            outputs = model(users, items).squeeze()

            loss = criterion(outputs, ratings)
            total_loss += loss.item()

    return total_loss / len(val_loader)

# Function to apply early stopping and track loss
def train_with_early_stopping(model, criterion, optimizer, train_loader, val_loader, config, val_triplets, device, patience=5):
    best_rmse = float('inf')
    epochs_no_improve = 0
    early_stop = False
    train_losses = []
    val_losses = []
    
    start_time = time.time()

    for epoch in range(config['epochs']):
        if early_stop:
            print("Early stopping...")
            break
        
        epoch_start_time = time.time()

        # Train and validate for one epoch
        train_loss = train_epoch(model, criterion, optimizer, train_loader, device)
        val_loss = evaluate_epoch(model, criterion, val_loader, device)

        # Track time for each epoch
        epoch_time = time.time() - epoch_start_time

        # Store losses for analysis
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Validation RMSE and Accuracy calculation
        full_matrix = predict_full_matrix(model, config['num_users'], config['num_items'], device)
        rmse, accuracy = evaluate_from_matrix(full_matrix, val_triplets)  # Using validation set

        # Print all relevant metrics for the epoch
        print(f"Epoch {epoch+1}/{config['epochs']}, Time: {epoch_time:.2f}s, "
              f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Validation RMSE: {rmse:.4f}, Validation Accuracy: {accuracy:.4f}")

        # Early stopping logic
        if rmse < best_rmse:
            best_rmse = rmse
            epochs_no_improve = 0
            # Save best model state_dict
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            early_stop = True

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    
    return train_losses, val_losses

# Predict full matrix
def predict_full_matrix(model, num_users, num_items, device):
    model.eval()
    full_matrix = np.zeros((num_users, num_items))
    with torch.no_grad():
        for user in range(num_users):
            users_batch = torch.full((num_items,), user, dtype=torch.long).to(device)
            items_batch = torch.arange(0, num_items, dtype=torch.long).to(device)
            predictions = model(users_batch, items_batch).squeeze().cpu().numpy()
            full_matrix[user, :] = predictions
    return full_matrix

# Updated function to evaluate RMSE and Accuracy using a mask from the test set
def evaluate_from_matrix(full_matrix, test_triplets):
    users = test_triplets[:, 0].astype(int)
    items = test_triplets[:, 1].astype(int)
    true_ratings = test_triplets[:, 2]

    # Apply the mask to only include non-zero or non-NaN ratings from test set
    mask = true_ratings > 0  # Assuming non-zero values are actual ratings

    # Extract the corresponding predicted and true ratings
    predicted_ratings = full_matrix[users, items][mask]
    true_ratings_filtered = true_ratings[mask]

    # Compute RMSE only for non-zero values
    rmse = np.sqrt(mean_squared_error(true_ratings_filtered, predicted_ratings))

    # Round both true and predicted ratings for accuracy calculation
    predicted_rounded = np.round(predicted_ratings)
    true_ratings_rounded = np.round(true_ratings_filtered)
    accuracy = accuracy_score(true_ratings_rounded, predicted_rounded)

    return rmse, accuracy