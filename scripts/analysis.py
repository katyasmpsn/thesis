import pandas as pd

df = pd.read_csv("results/clusters_notes_100_20_01_24")

def str_cat(x):
    """
    Helper function. Concats strings into readable list in a groupby func
    """
    return x.str.cat(sep=", ")


examples = df.groupby("Cluster").agg({"Word_Type": list, "Weights": "count"})


print(examples.iloc[1]['Word_Type'][:100])





#TODO: top N words (ranking)

#TODO: NPMI ranking