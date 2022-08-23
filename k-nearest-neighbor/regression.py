from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

import seaborn as sns


def main():
    """
    :resource: [RealPython.com](https://realpython.com/knn-python/)
    -
    - Goal is to predict the age of an abalone based on its physical measurements
    -
    :return:
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/abalone/abalone.data"
    )
    # Import the dataset
    abalone = pd.read_csv(url, header=None)
    print(abalone.head())

    abalone.columns = [
        "Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight",
        "Rings",
    ]

    # We want to use only measurements to predict the age of the abalone, since sex is not a physical measurement
    # we should remove it
    abalone = abalone.drop("Sex", axis=1)

    # Target variable is "Rings"
    abalone["Rings"].hist(bins=15)
    plt.show()

    # Determine correlation between features and rings
    correlation_matrix = abalone.corr()
    print(correlation_matrix["Rings"])

    # Define what "nearest" means
    X = abalone.drop("Rings", axis=1)
    # Convert to a numpy array
    X = X.values

    y = abalone["Rings"]
    y = y.values

    new_data_point = np.array([
        0.569552,
        0.446407,
        0.154437,
        1.016849,
        0.439051,
        0.222526,
        0.291208,
    ])

    distances = np.linalg.norm(X - new_data_point, axis=1)

    # Need to find the closest 3 neighbors
    k = 3
    nearest_neighbor_ids = distances.argsort()[:k]
    print("Nearest Neighbor IDs:", nearest_neighbor_ids)

    nearest_neighbor_rings = y[nearest_neighbor_ids]
    print("Nearest Neighbors:", nearest_neighbor_rings)
    # Determine prediction
    # For Regression, use Average
    # For Classification, use Mode

    prediction = nearest_neighbor_rings.mean()
    print("Final prediction:", prediction)

    # EXAMPLE FOR CLASSIFICATION
    class_neighbors = np.array(["A", "B", "B", "C"])
    mode = scipy.stats.mode(class_neighbors)
    print("Mode:", mode)

    # Deprecated: scipy.stats.mode(class_neighbors)
    # Use pandas.DataFrame.mode()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
    knn_model = KNeighborsRegressor(n_neighbors=3)
    knn_model.fit(X_train, y_train)

    # There are many evaluation metrics available for regression,
    # but you’ll use one of the most common ones,
    # the root-mean-square error (RMSE). The RMSE of a prediction is computed as follows:
    #
    # 1. Compute the difference between each data point’s actual value and predicted value.
    # 2. For each difference, take the square of this difference.
    # 3. Sum all the squared differences.
    # 4. Take the square root of the summed value.

    train_preds = knn_model.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("train Root-mean-square-error:", rmse)

    test_preds = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print("test Root-mean-square-error:", rmse)

    cmap = sns.cubehelix_palette(as_cmap=True)
    f, ax = plt.subplots()
    points = ax.scatter(
        X_test[:, 0], X_test[:, 1], c=test_preds, s=50, cmap=cmap
    )

    f.colorbar(points)
    plt.show()

    cmap = sns.cubehelix_palette(as_cmap=True)
    f, ax = plt.subplots()
    points = ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cmap
    )
    f.colorbar(points)
    plt.show()

    # GridSearchCV is a tool that is often used for tuning hyperparameters

    parameters = {"n_neighbors": range(1, 50)}
    gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
    gridsearch.fit(X_train, y_train)

    print("Best params:", gridsearch.best_params_)

    train_preds_grid = gridsearch.predict(X_train)
    train_mse = mean_squared_error(y_train, train_preds_grid)
    train_rmse = sqrt(train_mse)
    test_preds_grid = gridsearch.predict(X_test)
    test_mse = mean_squared_error(X_test, test_preds_grid)
    test_rmse = sqrt(test_mse)
    print("train rmse:", train_rmse)
    print("test rmse:", test_rmse)






if __name__ == "__main__":
    main()
