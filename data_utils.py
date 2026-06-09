import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def get_narma10(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 0.5, n)
    y = np.zeros(n)
    for t in range(10, n):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-10:t]) + 1.5*u[t-10]*u[t-1] + 0.1
    y = (y - y.mean()) / y.std()
    return u.reshape(-1,1), y.reshape(-1,1)

def get_mackey_glass(n=5000, tau=17, delta_t=0.1, seed=42):
    np.random.seed(seed)
    n_points = n + 1000
    x = np.zeros(n_points)
    x[0] = 1.2
    for t in range(1, n_points):
        idx = t - tau
        if idx < 0:
            x_delayed = 0.0
        else:
            x_delayed = x[idx]
        x[t] = x[t-1] + delta_t * (0.2 * x_delayed / (1.0 + x_delayed**10) - 0.1 * x[t-1])
    x = x[1000:]  
    if np.std(x) > 1e-12:
        x = (x - x.mean()) / x.std()
    else:
        x = x + 1e-6 * np.random.randn(len(x))
        x = (x - x.mean()) / x.std()
    return x[:-1].reshape(-1,1), x[1:].reshape(-1,1)
def get_real_temp():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url)
    data = df['Temp'].values.astype(float)
    data = (data - data.mean()) / data.std()
    return data[:-1].reshape(-1,1), data[1:].reshape(-1,1)

def get_ar_baseline(train_y, test_y, lags=15):
    def create_lags(data, p):
        X, Y = [], []
        for i in range(p, len(data)):
            X.append(data[i-p:i].flatten())
            Y.append(data[i])
        return np.array(X), np.array(Y)
    # Обучаем на train_y
    X_tr, Y_tr = create_lags(train_y.flatten(), lags)
    model = LinearRegression().fit(X_tr, Y_tr)
    # Для теста создаём лаги из test_y
    X_te, Y_te = create_lags(test_y.flatten(), lags)
    pred = model.predict(X_te).reshape(-1, 1)
    return Y_te.reshape(-1, 1), pred

def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2)) / np.std(y_true)

def mae_metric(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)