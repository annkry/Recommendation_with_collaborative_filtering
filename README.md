# Introduction
The authors of this project include Anna Krysta, Sabina Askerova, and Xichen Zhang.

This project consists of three directories, each containing different techniques to solve the problem of predicting movie ratings when only a sparse ratings table is available:

- `ALS` : This directory includes two techniques: Alternating Least Squares (ALS) and a method that incorporates user and movie biases that are independent of the user-movie interaction.
- `Gradient_descent` : This directory includes methods such as Gradient Descent Matrix Factorization, Hybrid Matrix Factorization, Multi-Layer Perceptron Matrix Factorization, and PyTorch Gradient Descent Matrix Factorization.
- `Net` : This directory includes methods that use Neural Networks to solve the problem.

For each directory, a `README` file is provided to explain the running process. We also provide an analysis of the training data in the `explore_data.ipynb` file.

## References

1. Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix Factorization Techniques for Recommender Systems". *Computer*, vol. 42, no. 8, 2009, pp. 30-37. [DOI: 10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)

2. Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *arXiv preprint arXiv:1810.04805*, 2018. [Link](https://arxiv.org/abs/1810.04805)

3. He, Xiangnan, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. "Neural Collaborative Filtering". *Proceedings of the 26th International Conference on World Wide Web*, 2017, pp. 173-182. [Link](https://arxiv.org/abs/1708.05031)