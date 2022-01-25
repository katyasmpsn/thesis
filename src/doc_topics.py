

import pandas as pd
from ast import literal_eval
import numpy as np

centroids = pd.read_csv("results/notes_centroids.csv")
centroids = centroids.drop(columns=["Unnamed: 0"])

docs = pd.read_csv("data/cleaned_data.csv")

# tweet_embeddings = pd.read_csv("embeddings124/tweet_embeddings_jan24.csv")
note_embeddings = pd.read_csv("notes_embeddings_PCA.csv")
note_embeddings = note_embeddings.drop(columns=['Unnamed: 0', 'SqDist', 'Cluster', 'Weights'])

ex = docs.iloc[0]

temp = pd.DataFrame(literal_eval(ex['noteTextList']), columns = ["word"])

note_embeddings_local = note_embeddings.merge(temp, how="right", left_on = "Word_Type", right_on="word")

dist = np.linalg.norm(centroids-note_embeddings_local)