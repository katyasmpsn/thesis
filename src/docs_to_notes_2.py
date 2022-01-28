import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

centroids = pd.read_csv("results/both_centroids.csv")
centroids = centroids.drop(columns=["Unnamed: 0"])

docs = pd.read_csv("results/cleaned_data.csv")

embeddings = pd.read_csv("both_embeddings_PCA.csv")
embeddings = embeddings.drop(columns=['Unnamed: 0', 'SqDist', 'Cluster', 'Weights'])

embeddings = embeddings[~embeddings.isnull().any(axis=1)]

note_df = docs[["noteTextList", "noteId", "tweetId"]]

tweet_df = docs[["tweetTextList", "noteId", "tweetId"]]




def return_distribution(row, col, centroids):

    # example_line = row

    temp = pd.DataFrame(literal_eval(row[col]), columns = ["word"])


    embeddings_local = embeddings.merge(temp, how="right", left_on = "Word_Type", right_on="word")
    embeddings_local = embeddings_local.drop(columns=["Word_Type", "word"])

    # TODO: why would some of the mappings return a null?
    embeddings_local = embeddings_local[~embeddings_local.isnull().any(axis=1)]

    note_embeddings_local = embeddings_local.to_numpy()
    centroids = centroids.to_numpy()

    temp = cosine_similarity(centroids, note_embeddings_local)
    temp = temp.sum(axis=1)

    distr = softmax(temp)

    final = {A: B for A, B in zip(range(len(centroids)), distr)}
    return final


note_df['probs'] = note_df.apply(lambda x: return_distribution(x, 'noteTextList', centroids), axis=1)
tweet_df['probs'] = tweet_df.apply(lambda x: return_distribution(x, 'tweetTextList', centroids), axis=1)
