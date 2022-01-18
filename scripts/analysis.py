import pandas as pd

df = pd.read_csv("clusters_pca_500.csv")

def str_cat(x):
    """
    Helper function. Concats strings into readable list in a groupby func
    """
    return x.str.cat(sep=", ")


examples = df.groupby("Cluster").agg({"Word_Type": str_cat, "Weights": "count"})



#TODO: top N words (ranking)

#TODO: NPMI ranking