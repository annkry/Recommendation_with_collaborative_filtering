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
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None

    # Convert the rating matrices to sparse format
    @staticmethod
    def convert_to_sparse(ratings):
        rows, cols = np.where(~np.isnan(ratings))
        values = ratings[rows, cols]
        sparse_matrix = csr_matrix((values, (rows, cols)), shape=ratings.shape)
        return sparse_matrix
    
    def round_to_half_int(self, R):
        return np.round(R * 2) / 2

    # ALS function for matrix factorization
    def alternating_least_squares(self, R, max_r, k, mask):
        num_users, num_items = R.shape
        
        # Initialize user and item latent factor matrices with random values
        self.user_factors = np.random.rand(num_users, self.num_factors) * max_r / k 
        self.item_factors = np.random.rand(num_items, self.num_factors) * max_r / k 

        self.global_bias = np.mean(R[mask])
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)

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
        
        R_clean = np.where(np.isnan(R), 0.0, R)

        # Iteratively update user and item matrices
        for iteration in range(self.num_iterations):
            self.user_factors = least_squares_step(self.item_factors, R, self.lambda_reg)
            self.item_factors = least_squares_step(self.user_factors, R.T, self.lambda_reg)

            for u in range(num_users):
                user_mask = mask[u, :]
                if np.sum(user_mask) > 0:
                    self.user_bias[u] = np.sum(user_mask * (R_clean[u, :] - (self.global_bias + self.item_bias + np.dot(self.user_factors[u, :], self.item_factors.T)))) / np.sum(user_mask)

            for i in range(num_items):
                item_mask = mask[:, i]
                if np.sum(item_mask) > 0:
                    self.item_bias[i] = np.sum(item_mask * (R_clean[:, i] - (self.global_bias + self.user_bias + np.dot(self.user_factors, self.item_factors[i, :])))) / np.sum(item_mask)
    
    # Train the ALS model
    def train(self, ratings_train, m , k, mask):
        self.alternating_least_squares(ratings_train, m, k, mask)

    # Get the user and item factor matrices
    def get_factors(self):
        return self.user_factors, self.item_factors

    # Create a function to return the matrix table of user-item interactions
    def get_predicted_ratings(self):
        R_pred = np.dot(self.user_factors, self.item_factors.T)
        R_pred += self.global_bias + self.user_bias[:, np.newaxis] + self.item_bias[np.newaxis, :]

        return self.round_to_half_int(R_pred)