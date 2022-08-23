# K-Means

## Resources

### Follow Along
[RealPython.com](https://realpython.com/k-means-clustering-python/)

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