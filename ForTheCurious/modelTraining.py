import pandas as pd
from sklearn.linear_model import LogisticRegression
from featureEngineering import preprocessData
from sklearn import preprocessing
import pickle

trainDf = pd.read_csv("datasets/train.csv")
testDf = pd.read_csv("datasets/test.csv")

preprocessData(trainDf, testDf)

X_train = trainDf.drop('Survived', axis=1).values
X_train = preprocessing.scale(X_train)
y_train = trainDf['Survived'].values

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

pickle.dump(logreg, open('../model.pkl', 'wb'))