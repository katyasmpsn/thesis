import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# reading in embeddings
df = pd.read_csv("embeddings.csv", index_col=False)
# not sure why this column exists
df = df.drop(columns=["Unnamed: 0"])
word_type_list = df.columns.tolist()

# transpose for PCA
df = df.T
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

X["word_type"] = word_type_list
weights = pd.read_csv("vocab_counts.csv")
weights.columns = ["word_type", "tf"]
X = pd.merge(X, weights, how="left", on="word_type")

word_type = X["word_type"].to_list()
weights = X["tf"].to_list()
X = X.drop(columns = ["word_type", "tf"])

kmeans = KMeans(n_clusters=20, random_state=0)
kmeans.fit(X, sample_weight=weights)
clusters = kmeans.predict(X)

X["Cluster"] = clusters
X["Word_Type"] = word_type
X["Weights"] = weights

outfile = "clusters.csv"
of = open(outfile, 'w')
X[["Cluster","Word_Type","Weights"]].to_csv(of)
of.close()

