"""
We get the last hidden state from BERT for each word token, using the entire note/tweet/reply as the context window.
Then the embeddings are averaged over each word type.
"""

import torch
from transformers import BertTokenizerFast, BertModel, logging
import pandas as pd
import gc
import numpy as np
import itertools
import sys
logging.set_verbosity_error()
clean_data = sys.argv[1]
outfile = sys.argv[2]

#clean_data = "data/cleaned_data.csv"
#outfile = "results/debug_embeddings.csv"

def getTokenEmbeddings(t1, model, t):
    """
    INPUT: full text
    OUTPUT: dictionary with each tokens last hidden state. If the original token
    was broken down into subwords, the average over subword representations is
    returned

    {token : 1x768 vector}
    """

    # this is possibly bad coding, `t` and `model` were instantiated outside of this function
    # in the first code block
    tokens = t(t1, return_attention_mask=False, return_token_type_ids=False)
    words_ids = tokens.word_ids()
    if len(words_ids) > 512:
        print("wait why did it tokenize it like this???")

    encoded_input = t(t1, return_tensors="pt")
    output = model(**encoded_input)

    # Average subword representations
    # Generate dummies for words_ids, multiply by the tensor
    wi_d = pd.get_dummies(pd.Series(words_ids)).T
    reduced_states = torch.matmul(
        torch.from_numpy(wi_d.values.astype("float32")), torch.squeeze(output["last_hidden_state"])
    )

    words = t1.split()
    res = {words[i]: reduced_states[i] for i in range(len(words))}

    # Clean up objects
    del wi_d, reduced_states

    return res

def process_embeddings(lst):
    # Input: List of lists of words and embeddings
    # Output: List of words + big numpy array of embeddings

    word_lst = [list(x.keys()) for x in lst]
    word_lst = [y for x in word_lst for y in x]

    # List of numpy arrays
    np_arr_list = np.vstack(tuple([torch.stack(tuple(list(x.values())), dim=0).detach().numpy() for x in lst]))

    return word_lst, np_arr_list

clean_file = open(clean_data)
df = pd.read_csv(clean_file)
clean_file.close()

print(df.shape)
# omit sequences with more than 512 tokens according to
# https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf

df = df[df["noteTextList"].apply(lambda x: len(x) < 512)]
# chunking
# https://stackoverflow.com/questions/44729727/pandas-slice-large-dataframe-into-chunks

n = 240  #chunk row size
list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]
# def generateEmbeddings(small_df):
#     strings = small_df.noteText.tolist() # don't make a list , just get the embeddings for each df row?
#     embeddings = [getTokenEmbeddings(x) for x in strings]
#     return embeddings

#all_embeds = [] #2D list of chunk results
words_list = []

if 'np_embeds' in globals():
    del np_embeds

for i in range(len(list_df)):
# change me before moving to Patas for a full run!
#for i in range(2):
    print("chunk {0}/{1}".format
          (i, len(list_df)))

    # Instantiate chunk-level bert model
    model = BertModel.from_pretrained("bert-base-uncased")
    t = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # define local version of get_token_embeddings
    def getTokenEmbeddingsLoc(t1):
        return getTokenEmbeddings(t1, model, t)

    # Extract embeddings from the words
    embeddings_loc = list_df[i]["noteText"].apply(lambda x: getTokenEmbeddingsLoc(x)).to_list()

    # Collapse embeddings into a numpy array
    names_loc, np_embeds_loc = process_embeddings(embeddings_loc)

    # Save names and append numpy array of embeddings
    words_list.extend(names_loc)
    try:
        np_embeds
    except NameError:
        np_embeds = np_embeds_loc
    else:
        np_embeds = np.vstack((np_embeds, np_embeds_loc))

    # Delete local bert model and tokenizer,
    # Collect garbage
    del model, t, embeddings_loc, np_embeds_loc
    gc.collect()

## Vocabulary list
print("The length of vocab is {}".format(len(words_list)))

# Put results into a Pandas dataframe at the very end
embed_varnames = ['emb_' + str(i) for i in range(768)]
df_embeds = pd.DataFrame(np_embeds, columns=embed_varnames)
df_embeds['word'] = words_list
cols_order = ['word', *embed_varnames]
df_embeds = df_embeds[cols_order]

# Generate a word count and a mean of the embeddings
word_count = df_embeds.groupby(['word'])[embed_varnames[0]].count()
word_count = word_count.sort_values(ascending=False)
df_embeds = df_embeds.groupby(['word'])[embed_varnames].mean()

## Vocabulary list
print("The length of the deduplicated vocab is {}".format(df_embeds.shape[0]))
print("The 30 most frequently occurring words are:")
print(word_count[0:30])

# outfile = "../results/embeddings.csv"
of = open(outfile, "w")
df_embeds.to_csv(of)
of.close()

gc.collect()