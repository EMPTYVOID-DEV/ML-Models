from pandas import read_csv
from sklearn.model_selection import train_test_split

data = read_csv("../datasets/emails.csv")

data = data.drop("Email No.", axis=1)

y = data["Prediction"]

x = data.drop("Prediction", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

setLenght = len(x_train)

prior = len(y_train[y_train.eq(0)])

spamMap = {}

nonSpamMap = {}

for column in x_train:
    spamMap[column] = len(x_train[x_train[column].ne(0) & y_train.eq(0)]) / prior
    nonSpamMap[column] = len(x_train[x_train[column].ne(0) & y_train.eq(1)]) / (
        setLenght - prior
    )


# make predictions


for idx, sample in x_test.iterrows():
    spam = prior / setLenght
    nonSpam = (setLenght - prior) / setLenght
    for key in spamMap:
        if sample[key] != 0:
            spam = spam * sample[key] * spamMap[key]
            nonSpam = nonSpam * sample[key] * nonSpamMap[key]
    print("Prediction for this sample : ")
    print(sample)
    print(spam > nonSpam)
