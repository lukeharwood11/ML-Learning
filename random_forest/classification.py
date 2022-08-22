import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def main():
    iris_dataset = datasets.load_iris()
    print(iris_dataset.target_names)
    print(iris_dataset.feature_names)

    data = pd.DataFrame({
        'sepal length': iris_dataset.data[:, 0],
        'sepal width': iris_dataset.data[:, 1],
        'petal length': iris_dataset.data[:, 2],
        'petal width': iris_dataset.data[:, 3],
        'species': iris_dataset.target
    })

    columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
    print(data.head())

    X = data[columns]
    y = data['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average=None))


if __name__ == "__main__":
    main()
