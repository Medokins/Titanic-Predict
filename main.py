import pandas as pd
from sklearn import svm
from featureEngineering import preprocessData

trainDf = pd.read_csv("datasets/train.csv")
testDf = pd.read_csv("datasets/test.csv")

trainDf = trainDf.drop("Ticket", axis=1)
trainDf = trainDf.drop("Fare", axis=1)
trainDf = trainDf.drop("Cabin", axis=1)
trainDf = trainDf.drop("Embarked", axis=1)
trainDf = trainDf.drop("PassengerId", axis=1)

preprocessData(trainDf)


Xtrain = trainDf.drop('Survived', axis=1).values
ytrain = trainDf['Survived'].values

clf = svm.SVC(kernel = 'linear')
clf.fit(Xtrain, ytrain)

testDf = testDf.drop("Ticket", axis=1)
testDf = testDf.drop("Fare", axis=1)
testDf = testDf.drop("Cabin", axis=1)
testDf = testDf.drop("Embarked", axis=1)

preprocessData(testDf)

predictionDict = {"PassengerId":[], "Survived": []}

for i in range(len(testDf)):
    predictionDict["PassengerId"].append(testDf["PassengerId"][i])
    predictionDict["Survived"].append(clf.predict([testDf.iloc[i][1:]])[0])
    # print("Passanger: ", testDf["PassengerId"][i], "Survived: ", clf.predict([testDf.iloc[i][1:]]))


predictionDataFrame = pd.DataFrame(predictionDict)

print(predictionDataFrame)

predictionDataFrame.to_csv("C:/Users/medok/OneDrive/Desktop/Titanic/Prediction.csv", index=False)

