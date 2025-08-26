import numpy as np
import pandas as pd

# =======================
# 1. Load & Preprocess Data
# =======================
df = pd.read_csv("Marriage_Divorce_DB.csv")

# Pisahkan fitur (X) dan target (y)
X = df.drop(columns=["Divorce Probability"]).values
y = df["Divorce Probability"].values.reshape(-1, 1)

# Normalisasi fitur
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X = (X - X_mean) / X_std

# Tambahkan bias
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Train-test split
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, test_idx = indices[:split], indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]


# =======================
# 2. SGD Optimizer
# =======================
def train_sgd(X, y, lr=0.1, epochs=500):
    m, n = X.shape
    theta = np.zeros((n, 1))
    losses = []

    for epoch in range(epochs):
        for i in range(m):
            rand_i = np.random.randint(m)
            xi = X[rand_i:rand_i+1]
            yi = y[rand_i:rand_i+1]
            error = xi.dot(theta) - yi
            grad = xi.T.dot(error)
            theta -= lr * grad
        loss = np.mean((X.dot(theta) - y) ** 2)
        losses.append(loss)
    return theta, losses


# =======================
# 2b. SGD Optimizer with L2 Regularization
# =======================
def train_sgd_l2(X, y, lr=0.01, epochs=500, l2=0.01):
    m, n = X.shape
    theta = np.zeros((n, 1))
    losses = []

    for epoch in range(epochs):
        for i in range(m):
            rand_i = np.random.randint(m)
            xi = X[rand_i:rand_i+1]
            yi = y[rand_i:rand_i+1]
            error = xi.dot(theta) - yi
            grad = xi.T.dot(error) + l2 * theta
            theta -= lr * grad
        loss = np.mean((X.dot(theta) - y) ** 2) + l2 * np.sum(theta ** 2)
        losses.append(loss)
    return theta, losses


# =======================
# 3. Mini-Batch Gradient Descent Optimizer
# =======================
def train_minibatch(X, y, lr=0.1, epochs=500, batch_size=16):
    m, n = X.shape
    theta = np.zeros((n, 1))
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            error = xi.dot(theta) - yi
            grad = xi.T.dot(error) / batch_size
            theta -= lr * grad
        loss = np.mean((X.dot(theta) - y) ** 2)
        losses.append(loss)
    return theta, losses


# =======================
# 3b. Mini-Batch Gradient Descent with L2 Regularization
# =======================
def train_minibatch_l2(X, y, lr=0.01, epochs=500, batch_size=16, l2=0.01):
    m, n = X.shape
    theta = np.zeros((n, 1))
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            error = xi.dot(theta) - yi
            grad = xi.T.dot(error) / batch_size + l2 * theta
            theta -= lr * grad
        loss = np.mean((X.dot(theta) - y) ** 2) + l2 * np.sum(theta ** 2)
        losses.append(loss)
    return theta, losses


# =======================
# 4. Adam Optimizer
# =======================
def train_adam(X, y, lr=0.1, epochs=1000, batch_size=16, beta1=0.9, beta2=0.999, eps=1e-8):
    m, n = X.shape
    theta = np.zeros((n, 1))
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            error = xi.dot(theta) - yi
            grad = xi.T.dot(error) / batch_size

            m_t = beta1 * m_t + (1 - beta1) * grad
            v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)

            m_hat = m_t / (1 - beta1 ** (epoch + 1))
            v_hat = v_t / (1 - beta2 ** (epoch + 1))

            theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        loss = np.mean((X.dot(theta) - y) ** 2)
        losses.append(loss)
    return theta, losses


# =======================
# 4b. Adam Optimizer with L2 Regularization
# =======================
def train_adam_l2(X, y, lr=0.01, epochs=1000, batch_size=16, beta1=0.9, beta2=0.999, eps=1e-8, l2=0.01):
    m, n = X.shape
    theta = np.zeros((n, 1))
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            error = xi.dot(theta) - yi
            grad = xi.T.dot(error) / batch_size + l2 * theta

            m_t = beta1 * m_t + (1 - beta1) * grad
            v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)

            m_hat = m_t / (1 - beta1 ** (epoch + 1))
            v_hat = v_t / (1 - beta2 ** (epoch + 1))

            theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        loss = np.mean((X.dot(theta) - y) ** 2) + l2 * np.sum(theta ** 2)
        losses.append(loss)
    return theta, losses


# =======================
# 5. Training & Evaluation
# =======================
theta_sgd, loss_sgd = train_sgd(X_train, y_train, lr=0.01, epochs=500)
theta_mb, loss_mb = train_minibatch(X_train, y_train, lr=0.01, epochs=500, batch_size=16)
theta_adam, loss_adam = train_adam(X_train, y_train, lr=0.01, epochs=500, batch_size=16)

# L2 regularization (lambda=0.01)
theta_sgd_l2, loss_sgd_l2 = train_sgd_l2(X_train, y_train, lr=0.01, epochs=500, l2=0.01)
theta_mb_l2, loss_mb_l2 = train_minibatch_l2(X_train, y_train, lr=0.01, epochs=500, batch_size=16, l2=0.01)
theta_adam_l2, loss_adam_l2 = train_adam_l2(X_train, y_train, lr=0.01, epochs=500, batch_size=16, l2=0.01)

# Evaluasi
def mse(X, y, theta):
    return np.mean((X.dot(theta) - y) ** 2)

print("=== SGD ===")
print("Train MSE:", mse(X_train, y_train, theta_sgd))
print("Test  MSE:", mse(X_test, y_test, theta_sgd))

print("\n=== Mini-Batch Gradient Descent ===")
print("Train MSE:", mse(X_train, y_train, theta_mb))
print("Test  MSE:", mse(X_test, y_test, theta_mb))

print("\n=== Adam Optimizer ===")
print("Train MSE:", mse(X_train, y_train, theta_adam))
print("Test  MSE:", mse(X_test, y_test, theta_adam))

print("\n=== SGD + L2 ===")
print("Train MSE:", mse(X_train, y_train, theta_sgd_l2))
print("Test  MSE:", mse(X_test, y_test, theta_sgd_l2))

print("\n=== Mini-Batch + L2 ===")
print("Train MSE:", mse(X_train, y_train, theta_mb_l2))
print("Test  MSE:", mse(X_test, y_test, theta_mb_l2))

print("\n=== Adam + L2 ===")
print("Train MSE:", mse(X_train, y_train, theta_adam_l2))
print("Test  MSE:", mse(X_test, y_test, theta_adam_l2))


# =======================
# 6. Analisis Underfit / Overfit
# =======================
def analyze(train_mse, test_mse):
    if abs(train_mse - test_mse) > 0.2:
        return "Model mengalami OVERFITTING"
    elif train_mse > 1 and test_mse > 1:
        return "Model mengalami UNDERFITTING"
    else:
        return "Model cukup seimbang"

print("\nAnalisis SGD:", analyze(mse(X_train, y_train, theta_sgd),
                               mse(X_test, y_test, theta_sgd)))
print("Analisis Mini-Batch:", analyze(mse(X_train, y_train, theta_mb),
                                      mse(X_test, y_test, theta_mb)))
print("Analisis Adam:", analyze(mse(X_train, y_train, theta_adam),
                               mse(X_test, y_test, theta_adam)))
print("Analisis SGD + L2:", analyze(mse(X_train, y_train, theta_sgd_l2),
                                    mse(X_test, y_test, theta_sgd_l2)))
print("Analisis Mini-Batch + L2:", analyze(mse(X_train, y_train, theta_mb_l2),
                                           mse(X_test, y_test, theta_mb_l2)))
print("Analisis Adam + L2:", analyze(mse(X_train, y_train, theta_adam_l2),
                                     mse(X_test, y_test, theta_adam_l2)))
