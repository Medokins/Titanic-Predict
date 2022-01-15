from hashlib import new
import pandas as pd

testDf = pd.read_csv("datasets/test.csv")
trainDf = pd.read_csv("datasets/train.csv")

def preprocessData(df):
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

#     Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex	
# Age	Age in years	
# sibsp	 of siblings / spouses aboard the Titanic	
# parch	 of parents / children aboard the Titanic	
# ticket	Ticket number	
# fare	Passenger fare	
# cabin	Cabin number	
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

    i = 0

    for name in df["Name"]:
        for title in titleList:
            if title in name:
                df.at[i, "Name"] = titleDict[title]
                break
        if df["Sex"][i] == "male":
            df.at[i, "Sex"] = 0
        elif df["Sex"][i] == "female":
            df.at[i, "Sex"] = 1


        if type(df["Age"][i]) != int:
            df.at[i, "Sex"] = 0
        

        i += 1