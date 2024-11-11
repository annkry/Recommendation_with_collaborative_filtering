
import numpy as np
import os
from tqdm import tqdm, trange
import argparse

import torch

from pytorch_gd_mf import MatrixFactorization, train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()

    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name)
    print('Ratings Loaded.')

    n_users, n_items = table.shape
    n_factors = 20   # K
    
    train_data = [(i, j, table[i, j]) for i, j in zip(*np.where(~np.isnan(table)))]
    train_data = np.array(train_data)
    
    model = MatrixFactorization(n_users, n_items, n_factors)
    train_model(model, train_data, n_epochs=50, batch_size=64, lr=0.01, lambda_U=0.1, lambda_I=0.1)
    
    model.eval()

    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(n_users), desc="Generating predictions"):
            user_idx = torch.LongTensor([i] * n_items)
            item_idx = torch.LongTensor(range(n_items))
            
            predictions = model(user_idx, item_idx).numpy()
            table[i, :] = predictions

    # Save the completed table 
    np.save("output.npy", table)
