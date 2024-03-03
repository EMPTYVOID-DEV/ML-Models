from pandas import read_csv, get_dummies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# age has no relation with the output y it correlation is -0.041
# since we have multi classes (A, B, C, X, Y) we need to use one-vs-rest


def splitClass(classNumber, data):
    return (data == classNumber).astype(int)


data = read_csv("../datasets/drug200.csv")
dataX = data.drop("Age", axis=1)
encoder = LabelEncoder()
x = get_dummies(dataX.drop("Drug", axis=1), ["Sex", "BP", "Cholesterol"])
y = encoder.fit_transform(dataX["Drug"])


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)


for i in range(0, 4):
    currentClassTrain = splitClass(yTrain, i)
    currentClassModel = DecisionTreeClassifier(random_state=42)
    currentClassModel.fit(xTrain, currentClassTrain)
    yPred = currentClassModel.predict(xTest)
    print("The prediction of module " + str(i), yPred)
    print("The accuracy of model " + str(i), accuracy_score(y_pred=yPred, y_true=yTest))

print(y)

print(yTest)
