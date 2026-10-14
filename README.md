
# PyTorch Linear Regression: Diabetes Progression Predictor

This is my very first end-to-end machine learning project built with PyTorch! It implements a custom training loop to predict disease progression using clinical patient data.

## 🚀 What This Project Does
* **Loads Real Medical Data:** Uses Scikit-learn's built-in diabetes dataset (442 patients, 10 clinical features like BMI, age, and blood pressure).
* **Data Preprocessing:** Applies `StandardScaler` to normalize features for stable gradient descent.
* **Custom Training Loop:** Implements a linear regression model using PyTorch (`nn.Linear`), Mean Squared Error loss (`nn.MSELoss`), and Stochastic Gradient Descent (`torch.optim.SGD`).
* **Visualizes Loss:** Tracks and plots the model's error reduction over 100 epochs using Matplotlib.

## 🛠️ Tech Stack
* Python
* PyTorch
* Scikit-Learn
* Matplotlib

## 📊 Results
The training loop successfully demonstrates gradient descent in action, showing a rapid drop
in MSE loss as the model optimizes its weights and bias over time.