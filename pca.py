# pca.py
import numpy as np 
import matplotlib.pyplot as plt

def pca(X=np.array([]), no_dims=2):
    print("Computing the Eigen Vectors which would serve as the new basis for the new dimensional data...")
    (n, d) = X.shape
    # normalization of the data
    X = X - np.tile(np.mean(X, 0), (n, 1))
    # eigen vector computation
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    # new feature values along the first two eigen vectors.
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

X = np.loadtxt("iris.txt")
labels = np.loadtxt("iris_label.txt")
Y=pca(X,2)
plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.title('PCA view of the dataset')
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.show()
