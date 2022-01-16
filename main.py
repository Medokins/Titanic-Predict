import pickle
import pandas as pd
import numpy as np
from ForTheCurious.featureEngineering import preprocessData

verbose = True

trainDf = pd.read_csv("ForTheCurious/datasets/train.csv")
testDf = pd.read_csv("ForTheCurious/datasets/test.csv")

preprocessData(trainDf, testDf)

model = pickle.load(open('model.pkl', 'rb')) #You can test different models here, and to see Your accuray submit Your prediction to Kaggle Titanic Competition
predictionDict = {"PassengerId":[], "Survived": []}

for i in range(len(testDf)):
    predictionDict["PassengerId"].append(np.int32(testDf["PassengerId"][i]))
    predictionDict["Survived"].append(np.int32(model.predict([testDf.iloc[i][1:]])[0]))
    if verbose:
        print("Passanger: ", testDf["PassengerId"][i], "Survived: ", np.int32(model.predict([testDf.iloc[i][1:]])))


predictionDataFrame = pd.DataFrame(predictionDict)
predictionDataFrame.to_csv("Prediction.csv", index=False)