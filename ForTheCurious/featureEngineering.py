import numpy as np
import re

def preprocessData(train_df, test_df):

    train_df.drop(['PassengerId'], axis=1, inplace=True)
    train_df.drop(["Ticket"], axis=1, inplace=True)
    test_df.drop(["Ticket"], axis=1, inplace=True)

    titles = {'Mrs': 2, 'Mr': 0, 'Master': 1, 'Miss': 2, 'Major': 1, 'Rev': 1,
                    'Dr':1, 'Ms':2, 'Mlle':2,'Col':1, 'Capt':1, 'Mme':2, 'Countess':3,
                    'Don':0, 'Jonkheer':0}

    genders = {"male": 0, "female": 1}

    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 0}
    data = [train_df, test_df]

    ports = {"S": 0, "C": 1, "Q": 2}
    commonValue = "S"

    i = 0

    for dataset in data:
        #using Cabin feature
        dataset['Cabin'] = dataset['Cabin'].fillna("U")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group()) #getting letter of the Deck
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(1)
        dataset['Deck'] = dataset['Deck'].astype(np.float)

        #using and filling NaN in age feature
        mean = train_df["Age"].mean()
        std = train_df["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        rand_age = np.random.randint(mean - std, mean + std, size = is_null)
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = train_df["Age"].astype(int)

        #filling Embarked
        dataset['Embarked'] = dataset['Embarked'].fillna(commonValue)
        dataset['Embarked'] = dataset['Embarked'].map(ports)

        #filling Fare
        dataset['Fare'] = dataset['Fare'].fillna(0)
        dataset['Fare'] = dataset['Fare'].astype(int)

        #using Name feature, converting name to Int
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].map(titles)
        dataset['Title'] = dataset['Title'].fillna(0)

        #using the Sex feature
        dataset['Sex'] = dataset['Sex'].map(genders)

        #creating categories for age feature
        dataset['Age'] = dataset['Age'].astype(int)
        dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

        #creating categories for Fare feature
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
        dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
        dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
        dataset['Fare'] = dataset['Fare'].astype(int)

        #adding new features:

        #Fare per Person
        dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['SibSp']+1)
        dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
        

    train_df.drop(['Cabin'], axis=1, inplace=True)
    test_df.drop(['Cabin'], axis=1, inplace=True)
    train_df.drop(['Name'], axis=1, inplace=True)
    test_df.drop(['Name'], axis=1, inplace=True)