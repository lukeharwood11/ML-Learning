import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    """
    :resource: https://realpython.com/k-means-clustering-python/
    :return:
    """
    features, true_labels = make_blobs(
        n_samples=200,
        centers=3,
        cluster_std=2.75,
        random_state=42
    )
    # StandardScaler - standardization that scales/shifts the values for
    # each numerical feature in the dataset so that the features have a mean of 0 and a standard deviation of 1
    # link for determining preprocessing techniques: https://scikit-learn.org/stable/modules/preprocessing.html
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # init controls the initialization technique, standard version of k-means algorithm sets it to "random"
    # setting it to "k-means++" employs an advanced trick to speed up convergence

    # n_clusters sets k for the clustering step **most important parameter**

    # n_init sets the number of initializations to perform. default is 10 runs

    # max_iter sets the number of maximum iterations for each initialization of the k-means algorithm

    kmeans = KMeans(
        init="random",
        n_clusters=3,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    # fit to the scaled features, performing 10 runs of the k-means algorithm on the data with a maximum of 300
    # iterations per run

    kmeans.fit(scaled_features)

    # statistics from the initialization are available as attributes after calling .fit()

    # lowest SSE value
    # SSE: the squared sum of the differences of each values from its cluster centroid.
    print("kmeans.inertia_:", kmeans.inertia_)

    # Final locations of the centroid
    print("kmeans.cluster_centers_:", kmeans.cluster_centers_)

    # The number of iterations required to converge
    print("kmeans.n_iter_:", kmeans.n_iter_)

    # cluster assignments are stored as a one-dimensional NumPy array in kmeans.labels_
    print("labels:", kmeans.labels_[:5])

    # Choosing the appropriate number of clusters
    # 1. The elbow method
    # 2. The silhouette coefficient

    # Elbow Method
    # Plot K vs SSE
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    # A list holding the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    # Plot sse vs k
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    # kneed is a tool that helps identify the elbow point
    kl = KneeLocator(
        range(1, 11), sse, curve="convex", direction="decreasing"
    )
    print("Elbow Point (by kneed): ", kl.elbow)

    # The silhouette coefficient is a measure of cluster cohesion and separation.
    # It quantifies how well a data point fits into its assigned cluster based on two factors:
    #
    # 1. How close the data point is to other points in the cluster
    # 2. How far away the data point is from points in other clusters

    # Silhouette coefficient values range between -1 and 1.
    # Larger numbers indicate that samples are closer to their clusters than they are to other clusters.

    silhouette_coefficients = []
    # You must start at 2 clusters for silhouette coefficients
    # (Otherwise an exception will be thrown)
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()



if __name__ == "__main__":
    main()
