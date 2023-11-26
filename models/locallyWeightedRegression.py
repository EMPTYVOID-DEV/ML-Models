import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# tau is metric that define how large the area that effects the predictions of h(x)
# tau decrease the area around the query point decrease
# try changing tau to increase/decrease the area


def locally_weighted_regression(X, y, query_point, tau=0.1):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]

    weights = np.exp(-((X - query_point) ** 2) / (2 * tau**2))

    W = np.diag(weights.flatten())

    theta = np.linalg.inv(X_b.T.dot(W).dot(X_b)).dot(X_b.T).dot(W).dot(y)

    query_point_b = np.array([1, query_point])
    prediction = query_point_b.dot(theta)

    return prediction


data = pd.read_csv("../datasets/Salary_Data.csv")

X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values.reshape(-1, 1)


query_point = 5.5


prediction = locally_weighted_regression(X, y, query_point, 0.2)


plt.scatter(X, y)
plt.scatter(query_point, prediction, color="red", marker="x", s=100, label="Prediction")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Locally Weighted Regression")
plt.legend()
plt.show()
