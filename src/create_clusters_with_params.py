import clustering
import random
import logging

num_clusters = [20]
dims_for_PCS = [768]
random_seeds = random.sample(range(0, 1000), 1)

wordsweights = clustering.getData()

temp_dfs = []
for d in dims_for_PCS:
    for n in num_clusters:
        for i, seed in enumerate(random_seeds):
            reduced_matrix, word_list, weight_list = clustering.PCACalc(wordsweights, d)
            temp, sil_score = clustering.KMeansCalc(
                reduced_matrix, word_list, weight_list, n, seed, i, d, write=True
            )
            logging.info(
                "K MEANS with {0} Clusters, {1} dims, random seed = {2} ({3}".format(
                    n, d, seed, i
                )
            )
            temp["numCluster"] = n
            temp["dimensions"] = d
            temp["seed"] = seed
            temp["silhouette_score"] = sil_score
            temp_dfs.append(temp)
