import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging


def getData():
    """

    :return: a dataframe with word types from the embeddings and their associated weights
    """
    in_file = "results/embeddings.csv"

    df = pd.read_csv(in_file, index_col=False)
    weights1 = pd.read_csv("results/tf_counts.csv")
    weights1.columns = ["word", "tf"]
    df = pd.merge(df, weights1, how="left", on="word")
    # # stop words arent in weights; will result in a nan
    df = df.dropna(subset = ["tf"])
    return df

"""
Step 1:
Initial PCA reduction. Sia et al found that KM allows for dimension reduction of up to 80%.
In Sia et al, this dimension reduction was primarily to reduce clustering complexity.
Not sure if the same logic applies for this exercise.
This should be tested as a hyperparameter; but we can start by going from 768 -> 100
"""

def PCACalc(df, dims):
    if dims != 768:
        logging.info("Using PCA to reduce to {} dimensions".format(dims))
        pca_100d = PCA(n_components=dims)
        word_type_list = df["word"].tolist()
        weights = df['tf'].tolist()
        X = pd.DataFrame(pca_100d.fit_transform(df.drop(columns=["word", "tf", "word"])))
    else:
        logging.info("Not using PCA! Using all {} dimensions from BERT".format(dims))
        word_type_list = df["word"].tolist()
        weights = df['tf'].tolist()
        X = df.drop(columns=["word", "tf"])
    return X, word_type_list, weights


"""
Step 2:
KMeans with N topics (centroids). Again, topics might be used a hyperparameter
Setting N = 20 for some initial testing
"""

def KMeansCalc(X, word_type_list, weights, n_clusters, seed, i, dim, write=False):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    kmeans.fit(X, sample_weight=weights)
    clusters = kmeans.predict(X)

    # Centroids

    # squared distance to cluster center
    X_dist = kmeans.transform(X)**2
    X['SqDist'] = X_dist.sum(axis=1).round(2)
    X["Cluster"] = clusters
    X["Word_Type"] = word_type_list
    X["Weights"] = weights

    if write:
        X.to_csv("results/test/beforererankandtop_{0}clusters_{1}dims_{2}seed.csv".format(n_clusters, dim, i))

    # rerank top 100 with TF

    X = X.sort_values(["Cluster", "Weights"], ascending=False)

    grouped_df = X.groupby("Cluster")
    top = grouped_df.head(100).reset_index()

    # get top J according to min distance

    top = top.sort_values(["Cluster", "SqDist"], ascending=True)

    grouped_df = top.groupby("Cluster")
    top = grouped_df.head(10).reset_index()

    out_file = "results/test/clusters_{0}clusters_{1}dims_{2}seed".format(n_clusters, dim, i)

    if write:
        of = open(out_file, 'w')
        top[["Cluster","Word_Type","Weights", "SqDist"]].to_csv(of)
        of.close()

    cluster_file = "results/test/centroids_{0}clusters_{1}dims_{2}seed.csv".format(n_clusters, dim, i)
    centroids = kmeans.cluster_centers_

    if write:
        cl = open(cluster_file, "w")
        pd.DataFrame(centroids).to_csv(cl)
        cl.close()

    return top[["Cluster","Word_Type","Weights", "SqDist"]]