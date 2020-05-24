import pandas as pd
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

# If we don't know how many clusters are there, then we should check score of the model 
# and decide the number of cluster after which increasing clusters does not increase score significantly.

clusters = range(1,10)
models = [KMeans(n_clusters=i) for i in clusters]
scores = [models[i].fit(X).score(X) for i in range(len(clusters))]
print(scores)

# So if we didn't know number of clusters before hand, then we would have made only 2 clusters.
# But Iris data has 3 clusters, so we will compare Kmeans predictions with 3 clusters.

model = KMeans(n_clusters=3, n_init=30).fit(scale(X))
predictions = model.labels_

# Let's plot actual and predicted clusters side by side, do not pay attention to the label but actual clustering
X["target"] = y
X["predict"] = predictions

fig, ax = plt.subplots(1,2)
sns.scatterplot(x="petal width (cm)", y="petal length (cm)", hue="target", ax=ax[0], data=X)
sns.scatterplot(x="petal width (cm)", y="petal length (cm)", hue="predict", ax=ax[1], data=X)
fig.show()
