import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

sns.pairplot(data=df, hue='target')
plt.show()

# Looks like we can almost create our target clusters using just petal width and petal length

X = df.drop(["sepal width (cm)","sepal length (cm)","target"], axis=1)
y = iris.target

model = KMeans(n_clusters=3, n_init=30).fit(scale(X))
predictions = model.labels_

# Let's plot actual and predicted clusters side by side, do not pay attention to the label but actual clustering
X["target"] = y
X["predict"] = predictions

fig, ax = plt.subplots(1,2)
sns.scatterplot(x="petal width (cm)", y="petal length (cm)", hue="target", ax=ax[0], data=X)
sns.scatterplot(x="petal width (cm)", y="petal length (cm)", hue="predict", ax=ax[1], data=X)
fig.show()
