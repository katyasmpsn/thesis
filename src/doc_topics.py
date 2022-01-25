

import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

centroids = pd.read_csv("results/notes_centroids.csv")
centroids = centroids.drop(columns=["Unnamed: 0"])

docs = pd.read_csv("data/cleaned_data.csv")

# tweet_embeddings = pd.read_csv("embeddings124/tweet_embeddings_jan24.csv")
note_embeddings = pd.read_csv("notes_embeddings_PCA.csv")
note_embeddings = note_embeddings.drop(columns=['Unnamed: 0', 'SqDist', 'Cluster', 'Weights'])

ex = docs.iloc[3]

temp = pd.DataFrame(literal_eval(ex['noteTextList']), columns = ["word"])

note_embeddings_local = note_embeddings.merge(temp, how="right", left_on = "Word_Type", right_on="word")
note_embeddings_local = note_embeddings_local.drop(columns=["Word_Type", "word"])

note_embeddings_local = note_embeddings_local.to_numpy()
centroids = centroids.to_numpy()

temp = cosine_similarity(centroids,note_embeddings_local)
temp = temp.sum(axis=1)

# res = []
# for center in range(len(centroids)):
#     center_res = []
#     for word in range(len(note_embeddings_local)):
#         dist = cosine_similarity(centroids[center],note_embeddings_local[word])
#         center_res.append(dist)
#     res.append(center_res)

distr = softmax(temp)