# ALS_X.py

import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import solve

class ALSModel:
    def __init__(self, num_factors=40, lambda_reg=8.0, num_iterations=50):
        self.num_factors = num_factors
        self.lambda_reg = lambda_reg
        self.num_iterations = num_iterations
        self.user_factors = None
        self.item_factors = None

    # Convert the rating matrices to sparse format
    @staticmethod
    def convert_to_sparse(ratings):
        rows, cols = np.where(~np.isnan(ratings))
        values = ratings[rows, cols]
        sparse_matrix = csr_matrix((values, (rows, cols)), shape=ratings.shape)
        return sparse_matrix

    # ALS function for matrix factorization
    def alternating_least_squares(self, R):
        num_users, num_items = R.shape
        
        # Initialize user and item latent factor matrices with random values
        self.user_factors = np.random.rand(num_users, self.num_factors)
        self.item_factors = np.random.rand(num_items, self.num_factors)

        # Function to solve the least squares problem
        def least_squares_step(fixed_factors, ratings, lambda_reg):
            num_fixed, num_factors = fixed_factors.shape
            updated_factors = np.zeros((ratings.shape[0], num_factors))  # Correct shape
            lambda_identity = lambda_reg * np.eye(num_factors)

            for i in range(ratings.shape[0]):
                idx = np.where(~np.isnan(ratings[i, :]))[0]  # Only consider non-NaN ratings
                if len(idx) == 0:
                    continue
                fixed_subset = fixed_factors[idx, :]
                ratings_subset = ratings[i, idx]
                updated_factors[i, :] = solve(fixed_subset.T.dot(fixed_subset) + lambda_identity, 
                                              fixed_subset.T.dot(ratings_subset))
            return updated_factors

        # Iteratively update user and item matrices
        for iteration in range(self.num_iterations):
            self.user_factors = least_squares_step(self.item_factors, R, self.lambda_reg)
            self.item_factors = least_squares_step(self.user_factors, R.T, self.lambda_reg)
    
    # Train the ALS model
    def train(self, ratings_train):
        self.alternating_least_squares(ratings_train)

    # Get the user and item factor matrices
    def get_factors(self):
        return self.user_factors, self.item_factors

    # Create a function to return the matrix table of user-item interactions
    def get_predicted_ratings(self):
        return np.dot(self.user_factors, self.item_factors.T)