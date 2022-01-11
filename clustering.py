import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# reading in embeddings
df = pd.read_csv("embeddings.csv")

"""
Step 1:
Initial PCA reduction. Sia et al found that KM allows for dimension reduction of up to 80%.
In Sia et al, this dimension reduction was primarily to reduce clustering complexity. 
Not sure if the same logic applies for this exercise. 
This should be tested as a hyperparameter; but we can start by going from 768 -> 100
"""

pca_100d = PCA(n_components=100)
X = pd.DataFrame(pca_100d.fit_transform(df))

"""
Step 2:
KMeans with N topics (centroids). Again, topics might be used a hyperparameter 
Setting N = 20 for some initial testing
"""

# based on https://www.kaggle.com/minc33/visualizing-high-dimensional-clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
clusters = kmeans.predict(X)
X["Cluster"] = clusters

