import pickle
import pandas as pd

testDf = pd.read_csv("ForTheCurious/datasets/test.csv")

model = pickle.load(open('model.pkl', 'rb'))
predictionDict = {"PassengerId":[], "Survived": []}

for i in range(len(testDf)):
    predictionDict["PassengerId"].append(testDf["PassengerId"][i])
    predictionDict["Survived"].append(model.predict([testDf.iloc[i][1:]])[0])
    # print("Passanger: ", testDf["PassengerId"][i], "Survived: ", clf.predict([testDf.iloc[i][1:]]))


predictionDataFrame = pd.DataFrame(predictionDict)
predictionDataFrame.to_csv("C:/Users/medok/OneDrive/Desktop/Titanic/Prediction.csv", index=False)