# k nearest neighbors
# git: @rviniciussouza
# git: @elmarkola

# read in the iris data
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()

# create X (features) and y (response)
X = iris.data[:,:2]
y = iris.target


# list(iris.target_names)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))