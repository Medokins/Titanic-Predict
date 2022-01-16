import pandas as pd
import numpy as np
import re

def preprocessData(train_df, test_df):

    train_df.drop(['PassengerId'], axis=1, inplace=True)
    train_df.drop(["Ticket"], axis=1, inplace=True)

    test_df.drop(["Ticket"], axis=1, inplace=True)

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
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data = [train_df, test_df]

    ports = {"S": 0, "C": 1, "Q": 2}
    commonValue = "S"

    i = 0

    for dataset in data:
        #using Cabin feature
        dataset['Cabin'] = dataset['Cabin'].fillna("U")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(np.float)

        #using and fillingNa in age feature
        mean = train_df["Age"].mean()
        std = test_df["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        rand_age = np.random.randint(mean - std, mean + std, size = is_null)
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = train_df["Age"].astype(int)

        #filling Embarked
        dataset['Embarked'] = dataset['Embarked'].fillna(commonValue)
        dataset['Embarked'] = dataset['Embarked'].map(ports)

    for name in train_df["Name"]:
        for title in titleList:
            if title in name:
                train_df.at[i, "Name"] = titleDict[title]
                break
        if train_df["Sex"][i] == "male":
            train_df.at[i, "Sex"] = 0
        elif train_df["Sex"][i] == "female":
            train_df.at[i, "Sex"] = 1

        if type(train_df["Fare"][i]) != float:
            train_df.at[i, "Fare"] = 0
        i += 1
    
    i = 0

    for name in test_df["Name"]:
        for title in titleList:
            if title in name:
                test_df.at[i, "Name"] = titleDict[title]
                break
        if test_df["Sex"][i] == "male":
            test_df.at[i, "Sex"] = 0
        elif test_df["Sex"][i] == "female":
            test_df.at[i, "Sex"] = 1

        if type(test_df["Fare"][i]) != float:
            test_df.at[i, "Fare"] = 0

            i += 1

    
    train_df.drop(['Cabin'], axis=1, inplace=True)
    test_df.drop(['Cabin'], axis=1, inplace=True)