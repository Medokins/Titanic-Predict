import pandas as pd
from sklearn.linear_model import LogisticRegression
from featureEngineering import preprocessData
import pickle

trainDf = pd.read_csv("datasets/train.csv")

trainDf.drop('PassengerId', axis=1, inplace=True)
trainDf.drop("Ticket", axis=1, inplace=True)
trainDf.drop("Cabin", axis=1, inplace=True)
trainDf.drop("Embarked", axis=1, inplace=True)

preprocessData(trainDf)
trainDf.dropna(inplace=True)

X_train = trainDf.drop('Survived', axis=1).values
y_train = trainDf['Survived'].values

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

pickle.dump(logreg, open('../model.pkl', 'wb'))