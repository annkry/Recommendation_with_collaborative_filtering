## Setup and Requirements

To install the required libraries, run:
  > pip install -r requirements.txt


## Usage

This repository provides two main scripts, `generate.py` and `generate_custom.py`, for completing ratings tables using various matrix completion models.

### `generate.py`

The `generate.py` script completes a ratings table and accepts the following argument:

- `--name`: The name of the `.npy` file containing the ratings matrix to be completed.

### `generate_custom.py`

The `generate_custom.py` script completes a ratings table using a specified matrix completion model and accepts the following arguments:

- `--name`: The name of the `.npy` file containing the ratings matrix to be completed.
- `--model`: The matrix completion model to use. Supported models are:
  - `gd_mf` (Gradient Descent Matrix Factorization)
  - `hybrid_mf` (Hybrid Matrix Factorization)
  - `mlp_mf` (Multi-Layer Perceptron Matrix Factorization)
  - `pytorch_gd_mf` (PyTorch Gradient Descent Matrix Factorization)

### Example Usage

To complete the `ratings_train.npy` matrix using the `gd_mf` model, run:
  > python generate_custom.py --name ratings_train.npy --model gd_mf

The script will process the input file and save the completed matrix as `output.npy` in the current directory.

## Evaluation of a Model
After generating the predicted ratings table, you can evaluate the model's performance using Root Mean Square Error (RMSE) and accuracy metrics. To perform the evaluation, ensure that `output.npy` (the completed ratings matrix) is in the current directory, and then run:
  > python evaluate_models.py --name ratings_test.npy

This command will compute and display the RMSE and accuracy for the predicted ratings table, allowing you to assess the model's performance.

## Model tests
To run tests for all available models, simply run:
  > chmod +x tests.sh  
  > ./tests.sh

The results of these tests will be saved in the `evaluation_output.txt` file. The metrics used here are RMSE and accuracy.