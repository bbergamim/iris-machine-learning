import platform
import pandas as pd
import numpy as np

os_name = platform.system()

if os_name == 'Windows': # Windows
    iris_data = pd.read_csv('D:/OneDrive/22. Python/Iris/Iris.csv')
elif os_name == 'Darwin': # MacOS
    iris_data = pd.read_csv('/Users/bbergamim/OneDrive/22. Python/Iris/Iris.csv')

iris_data['Species'] = iris_data['Species'].replace('Iris-setosa', 0)
iris_data['Species'] = iris_data['Species'].replace('Iris-versicolor', 1)
iris_data['Species'] = iris_data['Species'].replace('Iris-virginica', 2)

# Separate variables (predictors and target):
y = iris_data['Species']
x = iris_data.drop('Species', axis = 1)

# Eliminate Id (isn't a parameter):
x = iris_data.drop('Id', axis = 1)

from sklearn.model_selection import train_test_split

# Create try and test groups:
x_try, x_test, y_try, y_test = train_test_split(x, y, test_size = 0.3)

from sklearn.ensemble import ExtraTreesClassifier

# Create the model:
model = ExtraTreesClassifier()
model.fit(x_try, y_try)

# Accurace:
result = model.score(x_test, y_test)
result = "{:.2%}".format(result)
print("Accurace: ", result)

# Example (line 1 to 5):
n = []
original_ex = []
for n in y_test[1:5]:
    original_ex.append(n)

prediction = model.predict(x_test[1:5])
n = []
prediction_ex = []
for n in prediction:
    prediction_ex.append(n)

print('Example (original): ', original_ex)
print('Example (prediction): ', prediction_ex)
print(x_test[1:5])