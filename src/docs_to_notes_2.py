import pandas as pd
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import numpy as np

# reading in topic centroids
centroids = pd.read_csv("results/centroids_20clusters_768dims_0seed.csv")
centroids = centroids.drop(columns=["Unnamed: 0"])

# reading in the train data to map topics on to
docs = pd.read_csv("results/train.csv")

# reading in the word type embeddings
embeddings = pd.read_csv("results/beforererankandtop_20clusters_768dims_0seed.csv")
embeddings = embeddings.drop(columns=['Unnamed: 0', 'SqDist', 'Cluster', 'Weights'])

embeddings = embeddings[~embeddings.isnull().any(axis=1)]

note_df = docs[["noteTextList", "noteId", "tweetId"]]

tweet_df = docs[["tweetTextList", "noteId", "tweetId"]]



def return_distribution(row, col, centroids, top):

    # create a dataframe out of each documents words
    temp = pd.DataFrame(literal_eval(row[col]), columns = ["word"])

    # merge the document words with the token embeddings
    embeddings_local = embeddings.merge(temp, how="right", left_on = "Word_Type", right_on="word")
    # dims = [# tokens in doc x 768]
    embeddings_local = embeddings_local.drop(columns=["Word_Type", "word"])

    # TODO: why would some of the mappings return a null?
    embeddings_local = embeddings_local[~embeddings_local.isnull().any(axis=1)]

    # dims = [# tokens in doc x 768]
    note_embeddings_local = embeddings_local.to_numpy()

    # dims = [ # topics x 768]
    centroids = centroids.to_numpy()

    # dims = [# topics x # tokens in doc]
    cos = cosine_similarity(centroids, note_embeddings_local)

    if top > 0:
        # print("here")
        # set the cosine similarity to 0 for all but the N highest topics
        cos[cos.argsort(axis=0).argsort(axis=0) < top] = 0

    # dims = [ 1 x # topics]
    cos = cos.sum(axis=1)

    # dims = [ 1 x # topics]
    distr = softmax(cos)

    final = {A: B for A, B in zip(range(len(centroids)), distr)}
    return final

note_df['probs'] = note_df.apply(lambda x: return_distribution(x, 'noteTextList', centroids, top = 17), axis=1)
tweet_df['probs'] = tweet_df.apply(lambda x: return_distribution(x, 'tweetTextList', centroids, top = 17), axis=1)

note_df['doc'] = "notes"
tweet_df['doc'] = "tweets"

final = pd.merge(note_df, tweet_df, on=['noteId', 'tweetId'], indicator=True, suffixes=("_note", "_tweet"))

final.to_csv("distribution_top3.csv")