"""
This calculates term-frequency stats for each word type in the vocab.
At some point, it may contain more options
"""

import pandas as pd
from collections import Counter
from ast import literal_eval
import sys

input_type = sys.argv[1]

if input_type == "tweets":
    col_name = "tweetTextList"
    out_name = "results/tweet_vocab_counts.csv"
elif input_type == "notes":
    col_name = "noteTextList"
    out_name = "results/note_vocab_counts.csv"
else:
    print("Incorrect argument! Please use `tweets` or `notes`")

try:
    # TODO: update this to read in a filename that's passed in as an arg
    df = pd.read_csv("results/cleaned_data_jan24.csv")
    # when reading in the df, the list of words is evaluated as a string so we need literal eval to
    # make it into a list again
    df[col_name] = df[col_name].apply(literal_eval)
    vocab_counts = Counter()
    df[col_name].apply(vocab_counts.update)
    corpus_denominator = sum(vocab_counts.values())
    # tf weights
    vocab_counts = {
        key: value / corpus_denominator for (key, value) in vocab_counts.items()
    }
    vocab_counts = pd.DataFrame.from_dict(vocab_counts, orient="index", columns=["tf"])
    # write out to a file in the main dir
    vocab_counts.to_csv(out_name)

except:
    print("Couldn't find cleaned_data.csv. Check that preprocessing.py ran ")
