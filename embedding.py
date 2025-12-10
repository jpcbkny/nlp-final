from gensim.models import Word2Vec
from procrustes import smart_procrustes_align_gensim
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_col, theme_classic, scale_fill_manual, labs

# Read dem corpus from disc
with open("corpora/demText.txt") as f:
    demModel = f.read()

# Read rep corpus from disc
with open("corpora/repText.txt") as f:
    repModel = f.read()

demModel = Word2Vec(demSents, epochs=25, window=4)
repModel = Word2Vec(repSents, epochs=25, window=4)

demVec = demModel.wv
repVec = repModel.wv

repVecAligned = smart_procrustes_align_gensim(demModel, repModel)


def similarity_print(vec, target: str, subName: str, n: int = 10):

    print("Top " + str(n) + " words similar to " + target + " in " + subName + ":")

    for word, value in vec.most_similar(target, topn=n):

        print(f"{value: .2f} {word}")

TARGET = "taxes"

df = pd.DataFrame({
    "Community" : [],
    "Word" : [],
    "Similarity" : []
})

i = 0
for word, value in demVec.most_similar(TARGET, topn=50):
    df.iloc[i] = ["r/democrats", word, value]
    i += 1

for word, value in demVec.most_similar(TARGET, topn=50):
    df.iloc[i] = ["r/republicans", word, value]
    i += 1

p = (ggplot(df, aes(x="Word", y="Similarity", fill="Community")) +
 geom_col(position="dodge") +
 scale_fill_manual(values={"r/republicans" : "red", "r/democrats" : "blue"}) +
 labs(title='Semantic Similarity between "taxes" and Related Words') +
 theme_classic())

fig = p.draw()
fig.show()

similarity_print(demVec, "taxes", "democrats")
similarity_print(repVecAligned, "taxes", "republicans")
