import pandas as pd

testDf = pd.read_csv("datasets/test.csv")
trainDf = pd.read_csv("datasets/train.csv")

def preprocessData(trainDf):
    titleList = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']
    titleDict = {
    "Mrs": 2,
    "Mr": 1,
    "Master": 1,
    "Miss": 2,
    "Major": 1,
    "Rev": 2,
    "Dr": 1,
    "Ms": 1,
    "Mlle": 2,
    "Col": 1,
    "Capt": 1,
    "Mme": 2,
    "Countess": 3,
    "Don": 1,
    "Jonkheer": 1
    }

    i = 0

    for name in trainDf["Name"]:
        for title in titleList:
            if title in name:
                trainDf["Name"][i] = titleDict[title]
                break
        if trainDf["Sex"][i] == "male":
            trainDf["Sex"][i] = 0
        elif trainDf["Sex"][i] == "female":
            trainDf["Sex"][i] = 1

        if type(trainDf["Age"][i]) != int:
            trainDf["Age"][i] = 0
        i += 1