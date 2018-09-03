from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
iris_dataset = load_iris()

# description of IRIS dataset
print(iris_dataset["DESCR"])

# Iris Dataset 
print(iris_dataset["data"].shape)

# Feature Names
print("Feature Names:{}".format(iris_dataset['feature_names']))

X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], test_size=0.2, random_state=0)

#visulation of data

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', diagonal={'kde'},
hist_kwds={'bins': 20}, s = 100, alpha=.8)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(np.mean(y_pred == y_test))

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


