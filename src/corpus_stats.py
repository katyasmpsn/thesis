"""
This calculates term-frequency stats for each word type in the vocab.
At some point, it may contain more options
"""

import pandas as pd
from collections import Counter
from ast import literal_eval
import logging


def defineColumns(input_type):
    """
    :param input_type: specify if the stats should be for notes, tweets, or both
    :return: the dataframe, dataframe column, and ouput file
    """

    input_type = str(input_type)
    if input_type == "tweets":
        df = pd.read_csv("results/train.csv")
        col_list = "tweetTextList"
        col_text = "tweetText"
    elif input_type == "notes":
        df = pd.read_csv("results/train.csv")
        col_list = "notesTextList"
        col_text = "notesText"
    elif input_type == "both":
        col_list = "bothTextList"
        col_text = "bothText"

        # concating notes and tweets together to run corpus stats over both
        df = pd.read_csv("results/train.csv")
        dfn = df[["noteText", "noteTextList"]]
        dft = df[["tweetText", "tweetTextList"]]
        df = pd.concat(
            [
                dfn.rename(columns={"noteTextList": col_list, "noteText": col_text}),
                dft.rename(columns={"tweetTextList": col_list, "tweetText": col_text}),
            ],
            axis=0,
            ignore_index=True,
        )
        # TODO: should i randomize the order so pca doesn't implicitly pick up on the order? should i even use pca rip

    else:
        logging.debug("Incorrect argument! Please use `tweets` or `notes` or `both`")

    return df, col_list, col_text


def calculateTF(df, col_name):
    """

    :param df: input dataframe
    :param col_name: input column name, either referring to a column of notes/tweets/or both
    :param out_name: name of the file to be written out
    :return: writes out a two column csv with word type: term frequency
    """
    # when reading in the df, the list of words is evaluated as a string so we need literal eval to
    # make it into a list again
    df[col_name] = df[col_name].apply(literal_eval)
    vocab_counts = Counter()
    df[col_name].apply(vocab_counts.update)
    corpus_denominator = sum(vocab_counts.values())
    logging.info(
        " denominator for TF (ie. total token count) is {}".format(corpus_denominator)
    )
    # tf weights
    vocab_counts = {
        key: value / corpus_denominator for (key, value) in vocab_counts.items()
    }
    vocab_counts = pd.DataFrame.from_dict(vocab_counts, orient="index", columns=["tf"])
    # write out to a file in the results dir
    vocab_counts.to_csv("results/tf_counts.csv")
