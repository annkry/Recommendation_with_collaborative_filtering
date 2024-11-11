import numpy as np

class ALS:
    def __init__(self, r_0, r_1, seed, k, it_max, alpha, beta):
        np.random.seed(seed)
        self.k = k
        self.it_max = it_max
        self.alpha = alpha
        self.beta = beta
        self.I = None
        self.U = None
        self.I_start = np.random.rand(r_0, self.k)
        self.U_start = np.random.rand(r_1, self.k)

    def rmse(self, R, R_t, mask):
        return np.sqrt(np.sum((mask * (R - R_t)) ** 2) / np.sum(mask))

    def train(self, R, mask):
        I_t = self.I_start
        U_t = self.U_start
        
        for t in range(self.it_max):
            I_tp1 = np.matmul(np.matmul(mask * R, U_t), np.linalg.inv(np.matmul(np.transpose(U_t), U_t) + self.alpha * np.identity(U_t.shape[1])))
            U_tp1 = np.matmul(np.matmul(np.transpose(mask * R), I_t), np.linalg.inv(np.matmul(np.transpose(I_t), I_t) + self.beta * np.identity(I_t.shape[1])))

            I_t = I_tp1
            U_t = U_tp1

        self.U = U_t
        self.I = I_t
    
    def test(self, R, mask):

        R_t = np.matmul(self.I, np.transpose(self.U))

        return self.rmse(R, R_t, mask)