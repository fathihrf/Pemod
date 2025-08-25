import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # <-- Add this import

# Load dataset
df = pd.read_csv("Regression_age_of_marriage_data.csv")

# Data Cleaning
# Drop rows where target is missing
df = df.dropna(subset=["age_of_marriage"])

# Choose features and target
# We'll use gender, height, profession, and country as features for demonstration
X = df[["gender", "height", "profession", "country"]].copy()
y = df["age_of_marriage"].astype(float)

# Feature Engineering
# Height: convert to float inches
def parse_height(h):
    if pd.isnull(h):
        return np.nan
    try:
        feet, inches = h.split("'")
        feet = int(feet)
        inches = int(inches.replace('"', "")) if inches else 0
        return feet * 12 + inches
    except:
        return np.nan
X["height"] = X["height"].apply(parse_height)

# Fill missing values
X["height"] = X["height"].fillna(X["height"].mean())
X["gender"] = X["gender"].fillna("unknown")
X["profession"] = X["profession"].fillna("none")
X["country"] = X["country"].fillna("unknown")

# Encode categorical variables with simple one-hot encoding
X = pd.get_dummies(X, columns=["gender", "profession", "country"], drop_first=True)

# Normalize features
X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / (X_std + 1e-8)

# Convert to numpy arrays
X_np = X_norm.values
y_np = y.values

# Train-test split
np.random.seed(42)
idx = np.random.permutation(len(X_np))
split = int(0.8 * len(X_np))
train_idx, test_idx = idx[:split], idx[split:]
X_train, X_test = X_np[train_idx], X_np[test_idx]
y_train, y_test = y_np[train_idx], y_np[test_idx]

# Linear Regression with SGD, mini-batch, or ADAM
def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

class LinearRegressionSGD:
    def __init__(self, lr=0.01, epochs=900, batch_size=32, optimizer="sgd"):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.train_losses = []  # <-- Track training loss per epoch

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0

        # For ADAM
        m_w = np.zeros(n_features)
        v_w = np.zeros(n_features)
        m_b = 0
        v_b = 0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for epoch in range(self.epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)
            X, y = X[idx], y[idx]
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                y_pred = X_batch @ self.W + self.b
                error = y_pred - y_batch
                grad_W = 2 * X_batch.T @ error / len(y_batch)
                grad_b = 2 * np.sum(error) / len(y_batch)
                if self.optimizer == "sgd":
                    self.W -= self.lr * grad_W
                    self.b -= self.lr * grad_b
                elif self.optimizer == "adam":
                    # ADAM update
                    m_w = beta1 * m_w + (1 - beta1) * grad_W
                    v_w = beta2 * v_w + (1 - beta2) * (grad_W ** 2)
                    m_b = beta1 * m_b + (1 - beta1) * grad_b
                    v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
                    m_w_corr = m_w / (1 - beta1 ** (epoch + 1))
                    v_w_corr = v_w / (1 - beta2 ** (epoch + 1))
                    m_b_corr = m_b / (1 - beta1 ** (epoch + 1))
                    v_b_corr = v_b / (1 - beta2 ** (epoch + 1))
                    self.W -= self.lr * m_w_corr / (np.sqrt(v_w_corr) + eps)
                    self.b -= self.lr * m_b_corr / (np.sqrt(v_b_corr) + eps)
            # Track training loss
            y_pred_train = self.predict(X)
            train_loss = mse(y, y_pred_train)
            self.train_losses.append(train_loss)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1} Training MSE: {train_loss:.4f}")

    def predict(self, X):
        return X @ self.W + self.b

# Train model (choose optimizer: 'sgd' or 'adam')
model = LinearRegressionSGD(lr=0.05, epochs=900, batch_size=32, optimizer="adam")
model.fit(X_train, y_train)

# Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_mse = mse(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)

# Only plot points where both actual and predicted age > 0
mask = (y_test > 0) & (y_test_pred > 0)
y_test_pos = y_test[mask]
y_test_pred_pos = y_test_pred[mask]

plt.figure(figsize=(8, 5))
plt.scatter(y_test_pos, y_test_pred_pos, alpha=0.7, label="Test Data")
# Regression line (y = x)
if len(y_test_pos) > 0 and len(y_test_pred_pos) > 0:
    min_age = min(y_test_pos.min(), y_test_pred_pos.min())
    max_age = max(y_test_pos.max(), y_test_pred_pos.max())
    plt.plot([min_age, max_age], [min_age, max_age], color='red', linestyle='--', label="Ideal Fit (y=x)")
plt.xlabel("Actual Age of Marriage")
plt.ylabel("Predicted Age of Marriage")
plt.title("Actual vs Predicted Age of Marriage (Test Set, Age > 0)")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Underfit/Overfit Analysis
print("\nAnalysis:")
if abs(train_mse - test_mse) < 1.0:
    print("The model has similar training and test error; it is likely neither underfitting nor overfitting.")
elif train_mse < test_mse:
    print("The model has much lower training error than test error; it is likely overfitting.")
else:
    print("The model has high error on both train and test; it is likely underfitting.")

# Optional: show a few predictions
print("\nSample predictions:")
for i in range(5):
    print(f"Actual: {y_test[i]}, Predicted: {y_test_pred[i]:.2f}")
