import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys

input_type = sys.argv[1]
in_file = sys.argv[2]

# input_type = "tweets"
# in_file = "results/embeddings/tweet_embeddings_jan21.csv"

if input_type == "tweets":
    weight_file = "results/tweet_vocab_counts.csv"
elif input_type == "notes":
    weight_file = "results/note_vocab_counts.csv"
else:
    print("Incorrect argument! Please use `tweets` or `notes`")

# reading in embeddings
# df = pd.read_csv("results/embeddings/note_embeddings_jan21.csv", index_col=False)
df = pd.read_csv(in_file, index_col=False)
# not sure why this column exists
# df = df.drop(columns=["Unnamed: 0"])

word_type_list = df.word.tolist()

"""
Step 1:
Initial PCA reduction. Sia et al found that KM allows for dimension reduction of up to 80%.
In Sia et al, this dimension reduction was primarily to reduce clustering complexity.
Not sure if the same logic applies for this exercise.
This should be tested as a hyperparameter; but we can start by going from 768 -> 100
"""

dims = 100
pca_100d = PCA(n_components=dims)
X = pd.DataFrame(pca_100d.fit_transform(df.drop(columns=["word"]).reset_index()))

"""
Step 2:
KMeans with N topics (centroids). Again, topics might be used a hyperparameter
Setting N = 20 for some initial testing
"""

# based on https://www.kaggle.com/minc33/visualizing-high-dimensional-clusters

X["word_type"] = word_type_list
weights = pd.read_csv(weight_file)
weights.columns = ["word_type", "tf"]
X = pd.merge(X, weights, how="left", on="word_type")

word_type = X["word_type"].to_list()
weights = X["tf"].to_list()
X = X.drop(columns = ["word_type", "tf"])

n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X, sample_weight=weights)
clusters = kmeans.predict(X)

X["Cluster"] = clusters
X["Word_Type"] = word_type
X["Weights"] = weights

out_file = "results/clusters_{0}_{1}_{2}_{3}".format(input_type, dims, n_clusters, pd.Timestamp.today().strftime("%m_%d")

)
of = open(out_file, 'w')
X[["Cluster","Word_Type","Weights"]].to_csv(of)
of.close()

