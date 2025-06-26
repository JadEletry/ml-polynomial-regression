import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load polynomial regression data
train_data = pd.read_csv("polynomial_regression_train_data.csv")
test_data = pd.read_csv("polynomial_regression_test_data.csv")

X_train = train_data['x'].values.reshape(-1, 1)
y_train = train_data['y'].values
X_test = test_data['x'].values.reshape(-1, 1)
y_test = test_data['y'].values

st.title("Polynomial Regression Interactive Demo")

# Sidebar controls
degree = st.sidebar.slider("Polynomial Degree", min_value=1, max_value=12, value=3)

# Train model
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Smooth fitted curve generation
x_vals = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
y_vals = model.predict(x_vals)

# Plot results
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, label='Train Data', color='blue')
ax.scatter(X_test, y_test, label='Test Data', color='green')
ax.plot(x_vals, y_vals, label='Fitted Curve', color='red')
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Polynomial Regression (Degree={degree})")
st.pyplot(fig)

# Metrics
mse = np.mean((y_pred - y_test) ** 2)
st.write(f"**Mean Squared Error on Test Set:** {mse:.4f}")

# CSV Download
result_df = pd.DataFrame({"x": X_test.flatten(), "Actual y": y_test, "Predicted y": y_pred})
st.download_button("Download Predictions as CSV", result_df.to_csv(index=False), "polynomial_predictions.csv")
