import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from featureEngineering import preprocessData

trainDf = pd.read_csv("datasets/train.csv")
testDf = pd.read_csv("datasets/test.csv")

preprocessData(trainDf, testDf)

corrmat = trainDf.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

leaugeHeatMap=sns.heatmap(trainDf[top_corr_features].corr(),annot=True,cmap="RdYlGn")

plt.show()