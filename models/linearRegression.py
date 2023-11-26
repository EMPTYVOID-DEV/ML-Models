import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("../datasets/Salary_Data.csv")

X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values.reshape(-1, 1)

X_b = np.c_[np.ones((len(X), 1)), X]


def batch_gradient_descent(X, y, learning_rate=0.01, n_epochs=1000):
    m = len(y)
    theta = np.random.randn(2, 1)  # Random initialization

    for epoch in range(n_epochs):
        gradients = 2 / m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients

    return theta


theta_batch = batch_gradient_descent(X_b, y)

print("Theta from Batch Gradient Descent:", theta_batch)

plt.scatter(X, y)
plt.plot(X, X_b.dot(theta_batch), color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with Batch Gradient Descent")
plt.show()
