"""
This calculates term-frequency stats for each word type in the vocab.
At some point, it may contain more options
"""

import pandas as pd
from collections import Counter
from ast import literal_eval

try:
    # TODO: update this to read in a filename that's passed in as an arg
    df = pd.read_csv("../results/cleaned_data.csv")
    # when reading in the df, the list of words is evaluated as a string so we need literal eval to
    # make it into a list again
    df["noteTextList"] = df["noteTextList"].apply(literal_eval)
    vocab_counts = Counter()
    df["noteTextList"].apply(vocab_counts.update)
    corpus_denominator = sum(vocab_counts.values())
    # tf weights
    vocab_counts = {
        key: value / corpus_denominator for (key, value) in vocab_counts.items()
    }
    vocab_counts = pd.DataFrame.from_dict(vocab_counts, orient="index", columns=["tf"])
    # write out to a file in the main dir
    vocab_counts.to_csv("../results/vocab_counts.csv")

except:
    print("Couldn't find cleaned_data.csv. Check that preprocessing.py ran ")
