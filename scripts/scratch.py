import pandas as pd

df = pd.read_csv("./data/cleaned_data.csv")

# omit sequences with more than 512 tokens according to
# https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf
df['len'] = df['noteTextList'].apply(len)
print(df.nlargest(10, "len"))

df = df[df['noteTextList'].apply(lambda x: len(x) < 512)]


print(df.shape)


