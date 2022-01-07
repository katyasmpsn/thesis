"""
We get the last hidden state from BERT for each word token, using the entire note/tweet/reply as the context window.
Then the embeddings are averaged over each word type.
"""

import torch
from transformers import BertTokenizerFast, BertModel
import pandas as pd

model = BertModel.from_pretrained("bert-base-uncased")
t = BertTokenizerFast.from_pretrained("bert-base-uncased")


def getTokenEmbeddings(t1):
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

    res = {words[i]: reduced_states[i] for i in range(len(words))}
    return res


df = pd.read_csv("cleaned_data.csv")
small_df = df.head(100)
strings = small_df.noteText.tolist()
embeddings = [getTokenEmbeddings(x) for x in strings]

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


embeddings = d.values()
embeddings = [detach_embedding(x) for x in embeddings]

X = pd.DataFrame(embeddings)

X.to_csv("embeddings.csv")
