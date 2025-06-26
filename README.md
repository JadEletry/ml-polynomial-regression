# 📈 Polynomial Regression

This project presents an interactive visualization of **Polynomial Regression** using a simple dataset. Users can adjust the polynomial degree to observe how model complexity affects the regression curve and accuracy.

---

## 🎯 Features

- Upload-free app with built-in dataset
- Adjustable **polynomial degree** (1–12)
- Real-time visualization of fit over training & test data
- Mean Squared Error (MSE) reporting
- Option to download predictions as a CSV

---

## 🗂 Files Included

- `poly_app.py` – Streamlit app
- `polynomial_regression_train_data.csv` – Training dataset
- `polynomial_regression_test_data.csv` – Test dataset
- `requirements.txt` – Dependency file for deployment

---

## 📦 Run Locally

```bash
git clone https://github.com/JadEletry/ml-polynomial-regression-demo.git
cd ml-polynomial-regression-demo
pip install -r requirements.txt
streamlit run poly_app.py
```

---

## 🧠 About Polynomial Regression

Polynomial Regression is a form of linear regression in which the relationship between the independent variable `x` and the dependent variable `y` is modeled as an nth-degree polynomial. This is particularly useful for capturing non-linear trends in data.
