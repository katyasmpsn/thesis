import sys
import logging
import pandas as pd
import random
from sklearn.model_selection import train_test_split

import preprocessing
import corpus_stats
import generate_embeddings
import clustering

logging.basicConfig(filename="help.log", level=logging.INFO)
#
path_raw_notes = sys.argv[1]  # path to raw notes
path_raw_tweets = sys.argv[2]  # path to raw tweets
document_type = sys.argv[3]  # can be "tweets", "notes", or "both"
#
#
# # Write out cleaned, combined date to "results/cleaned_data.csv"
# preprocessing.preprocessMain(path_raw_notes, path_raw_tweets)
# logging.info("🎉 Successfully finished preprocessing Step")
#
# # Split into Train and Test sets
# clean_data = pd.read_csv("results/cleaned_data.csv")
# test_split = 0.99
# train, test = train_test_split(
#     clean_data, test_size=test_split, random_state=4, shuffle=True
# )
# logging.info(
#     "{0}/{1} split for train/test".format(((1 - test_split) * 100), test_split * 100)
# )
# train.to_csv("results/train.csv")
# test.to_csv("results/test.csv")
# logging.info("🎉 Successfully created test and train datasets")
#
# #   TRAIN: calculating corpus stats
# df_tf, col_wordList, col_wordText = corpus_stats.defineColumns(document_type)
# corpus_stats.calculateTF(df_tf, col_wordList)
# logging.info("🎉 Successfully created dataset from train data with Term-Frequencies")
#
# #  TRAIN: generating embeddings
#
# generate_embeddings.mainGeneration(df_tf, col_wordText)
# logging.info("🎉 Successfully generated embeddings!")
#
# # TRAIN: clustering the embeddings

num_clusters = [20, 50, 100]
dims_for_PCS = [100, 300, 500, 768]
random_seeds = random.sample(range(0, 1000), 3)

wordsweights = clustering.getData()

temp_dfs = []
for d in dims_for_PCS:
    for n in num_clusters:
        for i, seed in enumerate(random_seeds):
            reduced_matrix, word_list, weight_list = clustering.PCACalc(wordsweights, d)
            temp, sil_score = clustering.KMeansCalc(reduced_matrix, word_list, weight_list, n, seed, i, d)
            logging.info("K MEANS with {0} Clusters, {1} dims, random seed = {2} ({3}".format(
                n, d, seed, i
            ))
            temp['numCluster'] = n
            temp['dimensions'] = d
            temp['seed'] = seed
            temp['silhouette_score'] = sil_score
            temp_dfs.append(temp)

final = pd.concat(temp_dfs, axis=0, ignore_index=True)
final.to_csv("results/all_runs.csv")