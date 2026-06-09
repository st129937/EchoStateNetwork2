import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.linear_model import Ridge
import joblib
import os

class EchoStateNetwork:
    def __init__(self, n_inputs, n_res, n_outputs,
                 reservoir_type='random',   # 'random', 'orthogonal', 'cyclic'
                 rho=0.95, sparsity=0.1, leaking_rate=0.3,
                 input_scaling=0.1, beta=1e-8, seed=42):
        """
        reservoir_type:
          'random'       – разреженная случайная матрица
          'orthogonal'   – почти ортогональная: Q, затем масштабирование * rho
          'cyclic'       – простое кольцо: W_{i+1,i}=1, W_{1,N}=1, затем масштаб на rho
        """
        self.n_inputs = n_inputs
        self.n_res = n_res
        self.n_outputs = n_outputs
        self.rho = rho
        self.lr = leaking_rate
        self.beta = beta
        self.input_scaling = input_scaling
        self.reservoir_type = reservoir_type
        self.rng = np.random.RandomState(seed)

        # 1. Входная матрица (плотная) – общая для всех типов
        self.W_in = (self.rng.rand(n_res, 1 + n_inputs) - 0.5) * input_scaling

        # 2. Матрица резервуара в зависимости от типа
        if reservoir_type == 'random':
            W = sparse.random(n_res, n_res, density=sparsity, random_state=self.rng, format='csr')
            W.data -= 0.5
            current_rho = self._compute_spectral_radius(W)
            self.W = W * (rho / current_rho)

        elif reservoir_type == 'orthogonal':
            # Генерация случайной ортогональной матрицы через QR разложение
            H = self.rng.randn(n_res, n_res)
            Q, _ = np.linalg.qr(H)
            self.W = Q * rho      # масштабирование: спектральный радиус = rho

        elif reservoir_type == 'cyclic':
            # Простое кольцо: W_{i+1,i}=1, W_{1,N}=1
            W = np.zeros((n_res, n_res))
            for i in range(n_res-1):
                W[i+1, i] = 1.0
            W[0, n_res-1] = 1.0
            # масштабируем до нужного спектрального радиуса (у циклической матрицы все |λ|=1)
            self.W = W * rho

        else:
            raise ValueError(f"Unknown reservoir_type: {reservoir_type}")

        self.W_out = None
        self.last_state = np.zeros(n_res)

    def _compute_spectral_radius(self, matrix):
        if sparse.issparse(matrix):
            try:
                vals = eigs(matrix, k=1, which='LM', return_eigenvectors=False)
                return np.max(np.abs(vals))
            except:
                return np.max(np.abs(np.linalg.eigvals(matrix.toarray())))
        else:
            return np.max(np.abs(np.linalg.eigvals(matrix)))

    def _update(self, x, u):
        u_aug = np.concatenate([[1.0], u])
        pre_act = self.W_in @ u_aug + self.W @ x
        return (1 - self.lr) * x + self.lr * np.tanh(pre_act)

    def fit(self, inputs, targets, washout=100):
        states = []
        x = np.zeros(self.n_res)
        for t in range(len(inputs)):
            x = self._update(x, inputs[t])
            if t >= washout:
                states.append(np.concatenate([[1.0], inputs[t], x]))
        X = np.array(states)
        Y = targets[washout:]

        ridge = Ridge(alpha=self.beta, fit_intercept=False, solver='cholesky')
        ridge.fit(X, Y)
        self.W_out = ridge.coef_
        self.last_state = x

    def predict(self, inputs, x_init=None):
        x = x_init if x_init is not None else self.last_state
        preds = []
        for t in range(len(inputs)):
            x = self._update(x, inputs[t])
            feat = np.concatenate([[1.0], inputs[t], x])
            preds.append(self.W_out @ feat)
        return np.array(preds).reshape(-1, self.n_outputs)

    def save_matrices(self, folder="weights"):
        os.makedirs(folder, exist_ok=True)
        base = f"{folder}/{self.reservoir_type}"
        np.save(f"{base}_W_in.npy", self.W_in)
        if sparse.issparse(self.W):
            sparse.save_npz(f"{base}_W.npz", self.W)
        else:
            np.save(f"{base}_W.npy", self.W)