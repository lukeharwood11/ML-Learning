# K-Means
**Type**: Unsupervised Learning

K-means clustering is one of the simplest and popular unsupervised machine learning algorithms
## Resources

### Follow Along
[RealPython.com](https://realpython.com/k-means-clustering-python/)

### Other
[TowardsDataScience.com](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1#:~:text=K%2Dmeans%20clustering%20is%20one,known%2C%20or%20labelled%2C%20outcomes.)

## Notes

```
from sklearn.cluster import KMeans
```

### Determining K
You can use one of two methods

1. The Elbow Method
   - Visualize the bend of K vs SSE
   - _(SSE: the squared sum of the differences of each values from its cluster centroid._)
   - `kmeans.inertia_` = SSE
   
2. Silhouette Method
  - `from sklearn.metrics import silhouette_score`
  - Highest value is the best

```
    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()
```