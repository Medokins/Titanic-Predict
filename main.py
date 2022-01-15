import pickle
import pandas as pd
import numpy as np
from ForTheCurious.featureEngineering import preprocessData

verbose = True

testDf = pd.read_csv("ForTheCurious/datasets/test.csv")
testDf.drop("Ticket", axis=1, inplace=True)
testDf.drop("Cabin", axis=1, inplace=True)
testDf.drop("Embarked", axis=1, inplace=True)

preprocessData(testDf)
testDf = testDf.astype(float)

model = pickle.load(open('model.pkl', 'rb'))
predictionDict = {"PassengerId":[], "Survived": []}

for i in range(len(testDf)):
    predictionDict["PassengerId"].append(np.int32(testDf["PassengerId"][i]))
    predictionDict["Survived"].append(np.int32(model.predict([testDf.iloc[i][1:]])[0]))
    if verbose:
        print("Passanger: ", testDf["PassengerId"][i], "Survived: ", np.int32(model.predict([testDf.iloc[i][1:]])))


predictionDataFrame = pd.DataFrame(predictionDict)
predictionDataFrame.to_csv("Prediction.csv", index=False)