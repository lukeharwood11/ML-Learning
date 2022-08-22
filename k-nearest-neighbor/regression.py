import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from sklearn.model_selection import train_test_split

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
        "Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings",
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


if __name__ == "__main__":
    main()
