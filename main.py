import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_and_clean(csv_path):
	# Read CSV and convert European decimals/comma thousands
	df = pd.read_csv(csv_path)

	# Standardize numeric columns: replace commas used as decimal separators
	def clean_series(s):
		if s.dtype == object:
			s_clean = s.str.replace('.', '', regex=False)  # remove thousand sep if any
			s_clean = s_clean.str.replace(',', '.', regex=False)
			return pd.to_numeric(s_clean, errors='coerce')
		return pd.to_numeric(s, errors='coerce')

	df = df.apply(clean_series)
	return df


def combine_features(df):
	# Per request: "first two row combined variable" means first two columns combined as input features
	# We'll use them as a 2D feature vector: [col0, col1]
	X = df.iloc[:, :2].values
	Y = df.iloc[:, 2:].values  # remaining columns are targets (multi-output)
	feature_names = list(df.columns[:2])
	target_names = list(df.columns[2:])
	return X, Y, feature_names, target_names


def closed_form_linear_regression(X, y, l2=0.0):
	# y: (n_samples, ) or (n_samples, n_targets)
	n, d = X.shape
	Xb = np.hstack([np.ones((n, 1)), X])  # add bias
	I = np.eye(d + 1)
	I[0, 0] = 0  # don't regularize bias
	# closed-form solution: w = (X^T X + lambda I)^-1 X^T y
	A = Xb.T.dot(Xb) + l2 * I
	W = np.linalg.pinv(A).dot(Xb.T).dot(y)
	return W  # shape (d+1, n_targets) or (d+1,)


def predict_closed_form(W, X):
	n = X.shape[0]
	Xb = np.hstack([np.ones((n, 1)), X])
	return Xb.dot(W)


def mse_loss(y_true, y_pred):
	return mean_squared_error(y_true, y_pred)


def mae_loss(y_true, y_pred):
	return mean_absolute_error(y_true, y_pred)


def evaluate(y_true, y_pred):
	mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
	mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
	r2 = r2_score(y_true, y_pred, multioutput='raw_values')
	return mse, mae, r2


def plot_predictions(y_true, y_pred, target_names, out_dir):
	os.makedirs(out_dir, exist_ok=True)
	n_targets = y_true.shape[1]
	for i in range(n_targets):
		plt.figure(figsize=(6, 4))
		plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
		mn = min(y_true[:, i].min(), y_pred[:, i].min())
		mx = max(y_true[:, i].max(), y_pred[:, i].max())
		plt.plot([mn, mx], [mn, mx], 'r--')
		plt.xlabel('Actual')
		plt.ylabel('Predicted')
		plt.title(f'Parity plot - {target_names[i]}')
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, f'pred_vs_actual_{i}_{target_names[i].replace(" ", "_")}.png'))
		plt.close()


def main():
	base = os.path.dirname(__file__)
	csv_path = os.path.join(base, 'Dataset.csv')
	out_dir = os.path.join(base, 'outputs')
	os.makedirs(out_dir, exist_ok=True)

	print('Loading and cleaning data...')
	df = load_and_clean(csv_path)
	print('Data shape:', df.shape)

	X, Y, feature_names, target_names = combine_features(df)
	print('Features:', feature_names)
	print('Targets:', target_names)

	# Train/test split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

	# Closed-form OLS
	print('\nTraining closed-form linear regression (OLS)...')
	W = closed_form_linear_regression(X_train, Y_train, l2=0.0)
	Y_pred_cf = predict_closed_form(W, X_test)
	mse_cf, mae_cf, r2_cf = evaluate(Y_test, Y_pred_cf)
	print('Closed-form OLS per-target MSE:', mse_cf)
	print('Closed-form OLS per-target MAE:', mae_cf)
	print('Closed-form OLS per-target R2:', r2_cf)

	# Closed-form Ridge (L2 regularization)
	alpha = 1.0
	print(f'\nTraining closed-form Ridge (L2) with alpha={alpha}...')
	W_ridge = closed_form_linear_regression(X_train, Y_train, l2=alpha)
	Y_pred_ridge = predict_closed_form(W_ridge, X_test)
	mse_ridge, mae_ridge, r2_ridge = evaluate(Y_test, Y_pred_ridge)
	print('Closed-form Ridge per-target MSE:', mse_ridge)
	print('Closed-form Ridge per-target MAE:', mae_ridge)
	print('Closed-form Ridge per-target R2:', r2_ridge)

	# scikit-learn LinearRegression (multi-output)
	print('\nTraining scikit-learn LinearRegression...')
	lr = LinearRegression()
	lr.fit(X_train, Y_train)
	Y_pred_lr = lr.predict(X_test)
	mse_lr, mae_lr, r2_lr = evaluate(Y_test, Y_pred_lr)
	print('sklearn LinearRegression per-target MSE:', mse_lr)
	print('sklearn LinearRegression per-target MAE:', mae_lr)
	print('sklearn LinearRegression per-target R2:', r2_lr)

	# scikit-learn Ridge
	print(f'\nTraining scikit-learn Ridge with alpha={alpha}...')
	ridge = Ridge(alpha=alpha)
	ridge.fit(X_train, Y_train)
	Y_pred_sklearn_ridge = ridge.predict(X_test)
	mse_skr, mae_skr, r2_skr = evaluate(Y_test, Y_pred_sklearn_ridge)
	print('sklearn Ridge per-target MSE:', mse_skr)
	print('sklearn Ridge per-target MAE:', mae_skr)
	print('sklearn Ridge per-target R2:', r2_skr)

	# Loss/objective examples
	# For multi-output we compute average MSE
	def objective_mse_closed_form():
		return mse_loss(Y_test, Y_pred_cf)

	def objective_mse_ridge():
		# include regularization term for ridge
		# sum of squared weights excluding bias
		w = W_ridge[1:, :]
		reg = alpha * np.sum(w ** 2)
		return mse_loss(Y_test, Y_pred_ridge) + reg

	print('\nExample objective values:')
	print('Closed-form MSE (avg):', objective_mse_closed_form())
	print('Closed-form Ridge objective (MSE + L2):', objective_mse_ridge())

	# Save predictions plots
	plot_predictions(Y_test, Y_pred_lr, target_names, out_dir)

	# Save a small CSV with actual vs predicted (for first target only to keep small)
	compare_df = pd.DataFrame({
		'Actual_' + target_names[0]: Y_test[:, 0],
		'Predicted_' + target_names[0]: Y_pred_lr[:, 0]
	})
	compare_csv = os.path.join(out_dir, 'actual_vs_predicted_first_target.csv')
	compare_df.to_csv(compare_csv, index=False)

	print('\nSaved parity plots and CSV to', out_dir)


if __name__ == '__main__':
	main()
