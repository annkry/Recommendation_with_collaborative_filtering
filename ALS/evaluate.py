########## scipt for testing the model on a given set of ratings ##########
# to run use: python evaluate.py --name ratings_test.npy

import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_test.npy",
                        help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()

    table  = np.load("output.npy")

    R_test = np.load(args.name)

    mask = ~np.isnan(R_test)
    R_test = np.where(np.isnan(R_test), 0.0, R_test)
    
    test_rmse = np.sqrt(np.sum((mask * (R_test - table)) ** 2) / np.sum(mask))
    print(f"Test RMSE: {test_rmse:.4f}")

    def accuracy(R_true, R_pred, mask, threshold=0.5):
        diff = np.abs(R_true - R_pred)
        within_threshold = (diff <= threshold) * mask
        return np.sum(within_threshold) / np.sum(mask)

    test_accuracy = accuracy(R_test, table, mask)
    print(f"Test Accuracy: {test_accuracy:.4f}")