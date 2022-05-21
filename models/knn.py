from sklearn.neighbors import KNeighborsClassifier as knn


class KNN:
    def __init__(self, k=5):
        self.y = None
        self.x = None
        self.k = k
        self.knn = knn(k)

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.knn.fit(x, y)

    def predict(self, test):
        return self.knn.predict(test)

