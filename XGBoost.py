from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

# XGBoost need DMatrix instead of numpy arrays
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 4,
    'eta': 0.3,
    'objective': 'multi:softmax', # softmax=categorical, sigmoid=binary classification
    'num_class': 3} 
epochs = 20

model = xgb.train(param, train_data, epochs)
predictions = model.predict(test_data)

acc = metrics.accuracy_score(y_test, predictions)
print("Accuracy: {}".format(acc))