"""
Data Pipeline (it's ~manual~):
    * download notes and ratings data from https://twitter.github.io/birdwatch/contributing/download-data/
    * create a list of all unique tweet ids from the notes data, and save it to a .txt file
    * hydrate the tweets using DocNow hydrator (https://twitter.github.io/birdwatch/contributing/download-data/)
    * run this script!
"""

import pandas as pd
from nltk.tokenize import TweetTokenizer
import re
import string
from nltk.corpus import stopwords
import logging


def combineRaw(notes_filename, hydrated_tweets_filename):
    """
    :param notes_filename: path to raw Birdwatch notes information
    :param hydrated_tweets_filename: path to the note's tweets information
    :return: a dataframe with the combined note and tweet data, with
    """
    notes = open(notes_filename, "r")
    hydrated_tweets = open(hydrated_tweets_filename, "r")

    notes = pd.read_csv(notes, sep="\t")
    hydrated_tweets = pd.read_csv(hydrated_tweets)

    # the vast majority of birdwatch tweets seem to be in english, so to reduce the mapping
    # complexity; I'll look for only english language tweets too. In the future, a multilingual
    # option could be explored

    logging.info("Full raw tweet dataset size: {}".format(hydrated_tweets.shape))
    hydrated_tweets = hydrated_tweets[hydrated_tweets["lang"] == "en"]
    logging.info("Raw tweet with only English size: {}".format(hydrated_tweets.shape))

    # merging and re-naming datasets
    df = pd.merge(notes, hydrated_tweets, how="left", left_on="tweetId", right_on="id")

    df = df[["noteId", "tweetId", "classification", "summary", "text", "tweet_url"]]
    df.columns = [
        "noteId",
        "tweetId",
        "noteClassification",
        "noteText",
        "tweetText",
        "tweetURL",
    ]

    # omitting deleted tweets
    df = df[~df["tweetText"].isnull()]
    # omitting Birdwatch prompt tweets
    df = df[~df["noteText"].isnull()]

    return df


def textCleaning(rawtext):
    """
    Input: Raw text from notes or hydrated tweet
    Output: A string of lowercased tokens with usernames, hashtags, urls, stopwords, and digits omitted
    """
    tt = TweetTokenizer()

    tokens = tt.tokenize(rawtext)
    tokens = [x.lower() for x in tokens]

    hashtag_pattern = (
        r"\b[a-z]"  # first member would fail on hashtags, digits, usernames
    )
    # line below from https://stackoverflow.com/questions/6038061/regular-expression-to-find-urls-within-a-string
    url_pattern = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"

    tokens = [x for x in tokens if re.search(hashtag_pattern, x[0])]
    tokens = [x for x in tokens if not re.search(url_pattern, x)]

    # taking out all punctuation
    tokens = [s.translate(str.maketrans("", "", string.punctuation)) for s in tokens]

    # taking out all non-ascii letters
    printable = set(string.printable)
    tokens = ["".join(filter(lambda x: x in printable, s)) for s in tokens]

    # removing empty strings
    tokens = [i for i in tokens if i]

    return " ".join(tokens)


# TODO: WRITE ABOUT STOPWORDS. Omitting them from corpus stats but not as input for BERT


def removeStopwords(tokens):
    return [x for x in tokens if x not in stopwords.words("english")]


def createCleanFile(df):

    # These columns contain strings that have been cleaned; and will be fed into BERT
    df["noteText"] = df["noteText"].apply(textCleaning)
    df["tweetText"] = df["tweetText"].apply(textCleaning)

    # creating a column with a list of words for each text snippet so that it's easier to to calculate term frequencies over the corpus
    df["noteTextList"] = df["noteText"].str.lower().str.split()
    df["tweetTextList"] = df["tweetText"].str.lower().str.split()

    # The columns below will be used to calculate term frequencies and will be used as the main "doc"
    # to map the embeddings back on to
    df["noteTextList"] = df["noteTextList"].apply(removeStopwords)
    df["tweetTextList"] = df["tweetTextList"].apply(removeStopwords)

    # take out empty lists
    mask = (df["noteTextList"].str.len() > 0) & (df["tweetTextList"].str.len() > 0)
    df = df.loc[mask]

    # omit sequences with more than 512 tokens according to
    # https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf
    # max token length seems to be around 512

    # df['noteLength'] = df['noteTextList'].apply(lambda x: len(x))
    # df['tweetLength'] = df['tweetTextList'].apply(lambda x: len(x))

    return df


def preprocessMain(notes_filename, tweets_filename):

    dataset = combineRaw(notes_filename, tweets_filename)
    dataset = createCleanFile(dataset)
    # writing cleaned data to a file
    cleaned_data = open("results/cleaned_data.csv", "w")
    dataset.to_csv(cleaned_data)
    cleaned_data.close()

    del dataset
