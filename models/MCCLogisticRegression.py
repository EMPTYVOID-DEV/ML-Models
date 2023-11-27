import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Multi class classification with logistic regression--onevsrest method on iris dataset
# 1 for Setosa , 2 for Versicolor and 3 for Virginica
# wont work as intended for hyprid followers you need a model that can detect multi label classification

url = "../datasets/Iris.csv"
df = pd.read_csv(url)
X = df.drop("Species", axis=1)
X = X.drop("Id", axis=1)
Y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

y_setosa_train = (y_train == 1).astype(int)
y_versicolor_train = (y_train == 2).astype(int)
y_virginica_train = (y_train == 3).astype(int)

y_setosa_test = (y_test == 1).astype(int)
y_versicolor_test = (y_test == 2).astype(int)
y_virginica_test = (y_test == 3).astype(int)

setosa_model = LogisticRegression(random_state=42, max_iter=1000, tol=0.00001)
versicolor_model = LogisticRegression(random_state=42, max_iter=1000, tol=0.00001)
virginica_model = LogisticRegression(random_state=42, max_iter=1000, tol=0.0001)

setosa_model.fit(X_train, y_setosa_train)
versicolor_model.fit(X_train, y_versicolor_train)
virginica_model.fit(X_train, y_virginica_train)

setosa_prediction = setosa_model.predict(X_test)

versicolor_prediction = versicolor_model.predict(X_test)

virginica_prediction = virginica_model.predict(X_test)

print(y_test)

print(setosa_prediction)

print(versicolor_prediction)

print(virginica_prediction)

print("setosa accuracy", accuracy_score(y_setosa_test, setosa_prediction))

print("versicolor accuracy", accuracy_score(y_versicolor_test, versicolor_prediction))

print("virginica accuracy", accuracy_score(y_virginica_test, virginica_prediction))
