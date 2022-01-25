import pandas as pd

# import clusters
df = pd.read_csv("results/clusters_tweets_100_30_01_24"
                )

# clusters = df.groupby(["Cluster"]).agg({"Word_Type":list, "Weights": 'count'})

counts = pd.read_csv("results/tweet_vocab_counts.csv")

df = df.sort_values(["Cluster", "Weights"], ascending=False)

grouped_df = df.groupby("Cluster")
top = grouped_df.head(100).reset_index()

