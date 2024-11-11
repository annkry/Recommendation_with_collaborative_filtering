## Setup and Requirements

To install the required libraries, run:
  > pip install -r requirements.txt


## Usage

Use the file `generate.py` to complete your ratings table. 

It accepts the following arguments:
- `--name`: The name of the `.npy` file containing the ratings matrix to be completed.
- `--method`: The matrix completion method to use. You can specify either `als` or `als_bias` to generate the predicted ratings matrix.

### Example Usage

To complete the `ratings_train.npy` matrix using the `als_bias` method, run:
  > python3 generate.py --name ratings_train.npy --method als_bias

The script will process the input file and output the completed matrix as `output.npy`.

## Evaluation of a method
After generating the predicted ratings table, you can evaluate the model's performance using Root Mean Square Error (RMSE) and accuracy metrics. To perform the evaluation, ensure that  `output.npy` (the completed ratings matrix) is in the current directory, and then run:
  > python evaluate.py --name ratings_test.npy

This command will compute and display the RMSE and accuracy for the predicted ratings table, allowing you to assess the model's performance.