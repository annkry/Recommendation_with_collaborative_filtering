########## scipt for generating a filled ratings table on a given set of ratings ##########

import numpy as np
import argparse
from ALS_bias import ALSModel
from ALS import ALS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")
    parser.add_argument("--method", type=str, default="als_bias",
                        help="Name of the method")

    args = parser.parse_args()

    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name)
    print('Ratings Loaded.')

    if args.method == "als_bias":
        # ALS with user and movie bias
        R_train = np.where(np.isnan(table), 0.0, table)
        mask = ~np.isnan(table)

        als = ALSModel(num_factors=40, lambda_reg=8.0, num_iterations=50)
        als.train(table, np.max(R_train), 40, mask)

        predicted_table = als.get_predicted_ratings()
        table = predicted_table

    elif args.method == "als":
        # original ALS method
        R_train = np.where(np.isnan(table), 0.0, table)
        mask = ~np.isnan(table)

        als = ALS(r_0 = R_train.shape[0], r_1 = R_train.shape[1], seed = 48, k = 40, it_max = 50, alpha = 8.0, beta = 8.0)
        als.train(R_train, mask)

        predicted_table = als.get_predicted_ratings()

        table = predicted_table


    # Save the completed table 
    np.save("output.npy", table)
    print("Predicted ratings table is saved in the output.npy file.")