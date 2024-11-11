## Setup and Requirements

To install the required libraries, run:
  > pip install -r requirements.txt


## Usage

Use the file `generate.py` to complete your ratings table. 

It accepts the following arguments:
- `--name`: The name of the `.npy` file containing the ratings matrix to be completed.
- `--test`: The name of the `.npy` file containing the ratings table to test against.

### Example Usage

To complete the `ratings_train.npy` matrix using the Neural Network Matrix Factorization method, run:
  > python3 generate.py --name ratings_train.npy --test ratings_test.npy

The script will process the input file and output the completed matrix as `output.npy`. It will also calculate and print two metrics: RMSE and Accuracy on the testing ratings table.