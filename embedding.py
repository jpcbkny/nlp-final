#!/usr/bin/env python3

from gensim.models import Word2Vec
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

demModel = Word2Vec(demSents, epochs=25, window=4)
repModel = Word2Vec(repSents, epochs=25, window=4)

demVec = demModel.wv
repVec = repModel.wv


def similarity_print(vec, target: str, subName: str, n: int = 10):

    print("Top " + str(n) + " words similar to " + target + " in " + subName + ":")

    for word, value in vec.most_similar(target, topn=n):

        print(f"{value: .2f} {word}")


similarity_print(demVec, "taxes", "democrats")
similarity_print(repVec, "taxes", "republicans")
