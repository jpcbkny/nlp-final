#!/usr/bin/env python3

from gensim.models import Word2Vec
from procrustes import smart_procrustes_align_gensim
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_col, theme_classic, scale_fill_manual, labs, element_text, theme
from spacy.lang.en import English

nlp = English(pipeline=[])
nlp.add_pipe("sentencizer")

def get_sentences(text: str) -> list[list[str]]:
    """Split the specified text into sentences, consisting of text tokens."""

    sents = []

    # We process the text in chunks by paragraph, ensuring that a sentence
    # never crosses a paragraph boundary:
    for para in text.split("\n\n"):
        doc = nlp(para.replace("\n", " "))
        for sent in doc.sents:
            tokens = [
                token.text.lower().strip() for token in sent if not token.is_space
            ]
            sents.append(tokens)

    return sents

# Read and tokenize dem corpus from disc
with open("corpora/demText.txt") as f:
    demText = f.read()
    demSents = get_sentences(demText)

# Read and tokenize rep corpus from disc
with open("corpora/repText.txt") as f:
    repText = f.read()
    repSents = get_sentences(repText)

# TODO: ignore words with low frequencies
# TODO: find ideal threshold

demModel = Word2Vec(demSents, epochs=25, window=4)
repModel = Word2Vec(repSents, epochs=25, window=4)

print("Word2Vec models generated")

repModelAligned = smart_procrustes_align_gensim(demModel, repModel)

print("Embedding spaces aligned")

demVec = demModel.wv
repVec = repModelAligned.wv

print("")

def similarity_print(vec, target: str, subName: str, n: int = 10):

    print("Top " + str(n) + " words similar to " + target + " in " + subName + ":")

    for word, value in vec.most_similar(target, topn=n):

        print(f"{value: .2f} {word}")

TARGET = "taxes"



words = []
values = []
communities = []

for word, value in demVec.most_similar(TARGET, topn=10):
    words.append(word)
    values.append(value)
    communities.append("r/democrats")

for word, value in repVec.most_similar(TARGET, topn=10):
    words.append(word)
    values.append(value)
    communities.append("r/republicans")

df = pd.DataFrame({
    "Community" : communities,
    "Word" : words,
    "Similarity" : values
})


p = (ggplot(df, aes(x="Word", y="Similarity", fill="Community")) +
 geom_col(position="dodge") +
 scale_fill_manual(values={"r/republicans" : "red", "r/democrats" : "blue"}) +
 labs(title='Semantic Similarity between "taxes" and Related Words') +
 theme(axis_text_x=element_text(angle=45)))

p.save("taxes.png")

similarity_print(demVec, "taxes", "democrats")
similarity_print(repVec, "taxes", "republicans")
