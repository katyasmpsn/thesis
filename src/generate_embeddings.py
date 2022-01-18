"""
We get the last hidden state from BERT for each word token, using the entire note/tweet/reply as the context window.
Then the embeddings are averaged over each word type.
"""

import torch
from transformers import BertTokenizerFast, BertModel, logging
import pandas as pd
import gc
import itertools
import sys
logging.set_verbosity_error()
# clean_data = sys.argv[1]
# outfile = sys.argv[2]

clean_data = "data/cleaned_data.csv"
outfile = "results/debug_embeddings.csv"



def getTokenEmbeddings(t1):
    """
    INPUT: full text
    OUTPUT: dictionary with each tokens last hidden state. If the original token
    was broken down into subwords, the average over subword representations is
    returned

    {token : 1x768 vector}
    """

    model = BertModel.from_pretrained("bert-base-uncased")
    t = BertTokenizerFast.from_pretrained("bert-base-uncased")

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
    squeezed_states = torch.squeeze(output["last_hidden_state"])
    reduced_states = torch.matmul(
        torch.from_numpy(wi_d.values.astype("float32")), squeezed_states
    )
    words = t1.split()

    # Clean up objects
    del model, t, squeezed_states, wi_d

    res = {words[i]: reduced_states[i] for i in range(len(words))}
    return res


clean_file = open(clean_data)
df = pd.read_csv(clean_file)
clean_file.close()

print(df.shape)
# omit sequences with more than 512 tokens according to
# https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf

df = df[df["noteTextList"].apply(lambda x: len(x) < 512)]
# chunking
# https://stackoverflow.com/questions/44729727/pandas-slice-large-dataframe-into-chunks

n = 10  #chunk row size
list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]


# def generateEmbeddings(small_df):
#     strings = small_df.noteText.tolist() # don't make a list , just get the embeddings for each df row?
#     embeddings = [getTokenEmbeddings(x) for x in strings]
#     return embeddings

all_embeds = [] #2D list of chunk results
# for i in range(len(list_df)):
# change me before moving to Patas for a full run!
for i in range(2):
    print("chunk {0}/{1}".format
          (i, len(list_df)))

    list_df[i]["embeddings"] = list_df[i]["noteText"].apply(lambda x: getTokenEmbeddings(x))
    gc.collect()

    all_embeds.append(list_df[i])
#
# embeddings = list(itertools.chain(*all_embeds))

concat_results = pd.concat(all_embeds)
embeddings = concat_results["embeddings"].tolist()

print("The length of vocab is {}".format(len(embeddings)))

# just a toy function for now, but it's creating a master dictionary for the vocab.
# in the example: "look" is used twice, and is in two separate dictionaries in `text`.
# this adds all of the embeddings for "look" into one list
d = {}
for d_t in embeddings:
    for k, v in d_t.items():
        try:
            if d[k]:
                d[k].append(v)
        except KeyError:
            d[k] = [v]

# this now averages all of the embeddings for tokens that have more than one

for k, v in d.items():
    if len(v) > 1:
        d[k] = [torch.mean(torch.stack(v), dim=0)]

# dumb unit test; make me into a function
for key in d.keys():
    if len(d[key][0]) != 768:
        print("wrong size embedding?")

# detach takes out the grad requirement from the tensor


def detach_embedding(x):
    return x[0].detach().numpy()


d = {k: detach_embedding(v) for k, v in d.items()}
embeddings = pd.DataFrame.from_dict(d)

# outfile = "../results/embeddings.csv"
of = open(outfile, "w")
embeddings.to_csv(of)
of.close()
