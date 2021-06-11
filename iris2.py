from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()

'''
features = iris.data.T

sepal_lenght = features[0]
sepal_width = features[1]
petal_lenght = features[2]
petal_width = features[3]

sepal_lenght_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_lenght_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

# Generate graf showing relationship between sepal lenght and sepal width:
plt.scatter(sepal_lenght, sepal_width, c = iris.target)
plt.xlabel(sepal_lenght_label)
plt.ylabel(sepal_width_label)
plt.show()
'''

x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])

# prediction = knn.predict(x_new)
# print(prediction)

print(knn.score(x_test, y_test))