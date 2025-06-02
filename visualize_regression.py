import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the pre-trained models
# -----------------------------
with open('linear_regression_demo.mdl', 'rb') as file:
    linear_model = pickle.load(file)

with open('polynomial_regression_demo.mdl', 'rb') as file:
    poly_model = pickle.load(file)

# -----------------------------
# 2. Load the test data
# -----------------------------
x_test_linear = pd.read_csv('linear_test_x.csv')
x_test_poly = pd.read_csv('polynomial_test_x.csv')
y_test = pd.read_csv('linear_and_polynomial_test_y.csv')

# Convert to NumPy arrays to avoid compatibility warnings
x_test_linear = x_test_linear.values
x_test_poly = x_test_poly.values
y_test = y_test.values.flatten()

# -----------------------------
# 3. Make predictions with both models
# -----------------------------
y_pred_linear = linear_model.predict(x_test_linear)
y_pred_poly = poly_model.predict(x_test_poly)

# -----------------------------
# 4. Sort by X values to keep the plot smooth
# -----------------------------
sorted_idx = x_test_linear.flatten().argsort()
x_sorted = x_test_linear[sorted_idx]
y_pred_linear_sorted = y_pred_linear[sorted_idx]
y_pred_poly_sorted = y_pred_poly[sorted_idx]

# -----------------------------
# 5. Create the figure
# -----------------------------
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
plt.rcParams.update({'font.size': 24})

# Scatterplot of the actual data points
ax.scatter(x_sorted, y_test, c='red', label='Actual values')

# Plot the predictions
ax.plot(x_sorted, y_pred_linear_sorted, c='green', label='Linear Regression')
ax.plot(x_sorted, y_pred_poly_sorted, c='blue', label='Polynomial Regression')

# Titles and labels
ax.set_title('Linear and Polynomial Regression')
ax.set_xlabel('Input values (X)')
ax.set_ylabel('Predicted vs Actual (Y)')
ax.legend()

# -----------------------------
# 6. Save the plot as a JPG image
# -----------------------------
fig.savefig('Linear and Polynomial Visualisation Demo.jpg')
