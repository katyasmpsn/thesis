import pandas as pd
from ast import literal_eval
from collections import Counter
import numpy as np
import logging


def readData():
    """
    reads in the test data to create a word-word freq matrix and the top words from the clusters to calculate NPMI
    :return:
    """
    df = pd.read_csv("results/test.csv")
    clusters = pd.read_csv("results/all_runs.csv")
    # combining note and tweet text into a single list

    note_list = df.noteTextList.tolist()
    tweet_list = df.tweetTextList.tolist()

    note_list = [literal_eval(x) for x in note_list]
    tweet_list = [literal_eval(x) for x in tweet_list]

    word_list = note_list + tweet_list
    flat_word_list = [item for sublist in word_list for item in sublist]
    return word_list, flat_word_list, clusters, df


def defineWindow(test_data):
    test_data["noteLength"] = test_data["noteText"].str.split().str.len()
    test_data["tweetLength"] = test_data["tweetText"].str.split().str.len()
    tweetq1 = test_data["tweetLength"].quantile(0.25)
    noteq1 = test_data["noteLength"].quantile(0.25)
    logging.info(
        "The first quartile of note length {0}, the first quartile of tweet length {1}".format(
            noteq1, tweetq1
        )
    )
    window = noteq1  # will use this as the window for NPMI calculations as handwavy approximation of what could capture
    # semantics for tweets
    # TODO: research this!!!
    return int(window)


def omitMissingWords(cluster_df, test_word_list):
    """
    need a preprocessing step where if a word in the clusters isn't found in the test dataset, it's omitted
    because that word would have no frequency statistics associated with it
    :param cluster_df: dataframe with the top words in each cluster
    :param test_word_list: a 1D list of all words in the clusters
    :return: a smaller dataframe, where cluster words that aren't in the test set are omitted
    """
    cluster_df["Test_Train_Match"] = np.where(
        cluster_df["Word_Type"].isin(test_word_list), 1, 0
    )
    counts = cluster_df["Test_Train_Match"].value_counts()
    logging.info(
        "Distribution of cluster top words that are not in the test set", counts
    )
    cluster_df = cluster_df[cluster_df["Test_Train_Match"] == 1]

    return cluster_df


def return_neighborhood_cts(word_list, i, window):
    """
    Helper function to calculate NPMI statistics for the given window
    INPUT: a list of tokens, a pointer for the key word, window size
    OUTPUT: a dictionary of counts for the neighborhood of the key word
    """
    pre_pane_li = i - window
    pre_pane_ri = i
    post_pane_li = i + 1
    post_pane_ri = i + 1 + window
    if pre_pane_li < window and i < window:
        pre_pane_li = 0
    if i == len(word_list) - 1:
        pre_pane_li = len(word_list) - window - 1
    neighborhood = (
        word_list[pre_pane_li:pre_pane_ri] + word_list[post_pane_li:post_pane_ri]
    )
    inner_vecs = dict.fromkeys(set(neighborhood), 0)
    for word in neighborhood:
        inner_vecs[word] = inner_vecs[word] + 1
    return inner_vecs


def mainPMIStats(twod_word_list, window):
    """

    :param word_list: 2D list of words from the test dataset
    :param window:
    :return:
    """
    master_class = {}
    total_ct = 0
    for l in twod_word_list:
        word_list = l
        outer_vec = {}
        for i in range(len(word_list)):  # Looping thru all words
            tmp = return_neighborhood_cts(
                word_list, i, window
            )  # returning neighborhood counts for a key word
            if (
                word_list[i] in outer_vec.keys()
            ):  # if a key word already has a dict entry, add to it
                for k, v in tmp.items():
                    total_ct += v
                    try:
                        outer_vec[word_list[i]][k] = outer_vec[word_list[i]][k] + v
                    except:
                        outer_vec[word_list[i]][k] = v
            else:  # if a key word does not have a dict entry, create it
                outer_vec[word_list[i]] = tmp
                total_ct += sum(tmp.values())

        # updating the master dict over many documents
        for k, v in outer_vec.items():
            try:
                master_class[k] = Counter(master_class[k]) + Counter(outer_vec[k])
            except:
                master_class[k] = outer_vec[k]

    # make every dict in the master dictionary a counter object such that when a co-occurence doesn't occur, the joint
    # probability == 0

    master_class = {key: Counter(value) for key, value in master_class.items()}
    return master_class, total_ct


def npmi(w1, w2, vectors, vector_ct):
    # INPUT two words to compare, dict of vectors, total word count for denom
    # OUTPUT ppmi for those two words
    eps = 10 ** (-12)
    # numerator
    w1w2_dc = vectors[w1][w2] / vector_ct
    w1_dc = sum(vectors[w1].values()) / vector_ct
    w2_dc = sum(vectors[w2].values()) / vector_ct

    pmi_w1w2 = np.log((w1w2_dc) / ((w1_dc * w2_dc) + eps) + eps)
    npmi_w1w2 = pmi_w1w2 / (-np.log((w1w2_dc) + eps))
    return npmi_w1w2


def avgClusterNPMI(cluster_words, stats, total_ct):
    """
    Input: 1D List of words in a cluster
    Output: Average NPMI for the cluster
    wtf is happening? It is averaged twice: NPMI is calculated for each w1, w2; so an average of all the pairs is taken
    And, an average of all of the words is taken in the cluster.
    It looks like Lau et al sums the NPMIs but that would result in NPMIs that aren't really interpretable (not
    within the [-1,1] range)
    """

    npmi_scores = {}
    for w1 in cluster_words:
        # npmi_sum = 0
        # ct = 0
        for w2 in cluster_words:
            if w1 != w2:
                res = npmi(w1, w2, stats, total_ct)
                #                 print(w1,w2,res)
                #                 print(w1, w2, res)
                try:
                    if npmi_scores[(w1, w2)]:
                        print ("Collision! ")
                except:
                    npmi_scores[(w1,w2)] = res
                # npmi_sum += res
                # ct += 1

            else:
                pass
        # taking the average of every w1 against every other w2
        #  / ct

    res = 0
    for val in npmi_scores.values():
        res += val

    # using len() to get total keys for mean computation
    # averaging all words in a cluster

    res = res / len(npmi_scores)

    return res, npmi_scores


nested_cluster_words, flattened_cluster_words, cluster_df, test_df = readData()
window = defineWindow(test_df)
cluster_df = omitMissingWords(cluster_df, flattened_cluster_words)
wordword_freq, total_ct = mainPMIStats(nested_cluster_words, window)
npmi_score, word_scores = avgClusterNPMI(
    ["biden", "putin", "russia", "usa"], wordword_freq, total_ct
)
