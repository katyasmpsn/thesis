"""
Data Pipeline (it's ~manual~):
    * download notes and ratings data from https://twitter.github.io/birdwatch/contributing/download-data/
    * create a list of all unique tweet ids from the notes data, and save it to a .txt file
    * hydrate the tweets using DocNow hydrator (https://twitter.github.io/birdwatch/contributing/download-data/)
    * join the hydrated tweet information to the notes data
    * run this script!
"""

import pandas as pd
from nltk.tokenize import TweetTokenizer
import re
import sys

# The raw files are passed in from the command line
notes_filename = sys.argv[1]
hydrated_tweets_filename = sys.argv[2]
# cleaned_data_filename = sys.argv[3] # output file should always be named the same?

notes = open(notes_filename, "r")
hydrated_tweets = open(hydrated_tweets_filename, "r")

notes = pd.read_csv(notes, sep="\t")
hydrated_tweets = pd.read_csv(hydrated_tweets)

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

tt = TweetTokenizer()


def textCleaning(rawtext):
    """
    Input: Raw text from notes or hydrated tweet
    Output: A list of lowercased tokens with usernames, hashtags, urls, stopwords, and digits omitted
    """

    tokens = tt.tokenize(rawtext)
    tokens = [x.lower() for x in tokens]

    hashtag_pattern = (
        r"\b[a-z]"  # first member would fail on hashtags, digits, usernames
    )
    # line below from https://stackoverflow.com/questions/6038061/regular-expression-to-find-urls-within-a-string
    url_pattern = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"

    tokens = [x for x in tokens if re.search(hashtag_pattern, x[0])]
    tokens = [x for x in tokens if not re.search(url_pattern, x)]
    # TODO: check if Sia et al omitted stopwords here? On first read they did, but BERT would likely pick up on the
    # TODO: unnaturalness. Uncomment the line below if they should be omitted
    # TODO: but now i'm wondering if stop words will fuck with the corpus stats. ask shane
    # tokens = [x for x in tokens if x not in stopwords.words('english')]

    return " ".join(tokens)


df["noteText"] = df["noteText"].apply(textCleaning)
df["tweetText"] = df["tweetText"].apply(textCleaning)

# creating a column with a list of words for each text snippet so that it's easier to to calculate term frequencies over the corpus
df["noteTextList"] = df["noteText"].str.lower().str.split()
df["tweetTextList"] = df["tweetText"].str.lower().str.split()

# take out empty lists
mask = (df["noteTextList"].str.len() > 0) & (df["tweetTextList"].str.len() > 0)
df = df.loc[mask]

# omit sequences with more than 512 tokens according to
# https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf
# max token length seems to be around 512

# df['noteLength'] = df['noteTextList'].apply(lambda x: len(x))
# df['tweetLength'] = df['tweetTextList'].apply(lambda x: len(x))


# writing cleaned data to a file
cleaned_data = open("results/cleaned_data.csv", "w")
df.to_csv(cleaned_data)
cleaned_data.close()
