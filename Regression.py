from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Loaded preloaded dataset of Iris flowers available in sklearn
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=['sl','sw','pl','pw'])

sns.set()

# Let's see relationship among these features
sns.pairplot(df)
plt.show()

# Looks like a linear relationship between Petal Length(PL) and Petal Width(PW)
# This will shuffle while splitting, so the result will be different in every execution.
train_data, test_data = train_test_split(df[['pl','pw']], test_size=0.4)
slope, intercept, r_value, p_value, std_err = stats.linregress(train_data['pl'], train_data['pw'])
print("R2 Score in Linear Train Data: {}".format(r_value**2))

# Linear function to predict PW using PL
linear_func = lambda x: slope*x+intercept
r2_test = r2_score(test_data['pw'], linear_func(test_data['pl']))
print("R2 Score in Linear Test Data: {}".format(r2_test))

# Let's show actual and Predicted values
sample_x = np.arange(0,7,0.001)

sns.scatterplot(x='pl', y='pw', data=test_data)
sns.lineplot(sample_x, linear_func(sample_x), color='r')
plt.show()

# Now let's try polynomial regression betweeen Sepal Length(SL) and Petal Length(PL)
train_data, test_data = train_test_split(df[['pl','sl']], test_size=0.4)
predict = np.poly1d(np.polyfit(train_data['pl'], train_data['sl'], 3))
r2_test = r2_score(test_data['sl'], predict(test_data['pl']))
print("R2 Score in Polynomial Test Data: {}".format(r2_test))

# Let's show actual and Predicted values
sample_x = np.arange(0,7,0.001)

sns.scatterplot(x='pl', y='sl', data=test_data)
sns.lineplot(sample_x, predict(sample_x), color='r')
plt.show()