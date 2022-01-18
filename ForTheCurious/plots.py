from sklearn.metrics import precision_recall_curve
from featureEngineering import preprocessData
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd

trainDf = pd.read_csv("datasets/train.csv")
testDf = pd.read_csv("datasets/test.csv")

preprocessData(trainDf, testDf)

random_forest = LogisticRegression()
Xtrain = trainDf.drop("Survived", axis=1)
ytrain = trainDf["Survived"]
random_forest.fit(Xtrain, ytrain)

yscores = random_forest.predict_proba(Xtrain)
yscores = yscores[:,1]
precision, recall, threshold = precision_recall_curve(ytrain, yscores)

def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
#plot_precision_vs_recall(precision, recall)
plt.show()
r_a_score = roc_auc_score(ytrain, yscores)
print("ROC-AUC-Score:", r_a_score)
