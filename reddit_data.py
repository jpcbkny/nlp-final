#!/usr/bin/env python3

# import nltk 
# nltk.download('punkt')
from gensim.models import Word2Vec
from convokit import Corpus, download
demCorpus = Corpus(filename=download("subreddit-democrats"))
repCorpus = Corpus(filename=download("subreddit-republicans"))

from spacy.lang.en import English

nlp = English(pipeline=[])
nlp.add_pipe("sentencizer")

# creates text of r/democrats by going through every utterance in the subreddit and adding it with a newline to a string
demText = ""
demGenerator = demCorpus.iter_utterances()
for utterance in demGenerator:
    demText += "\n"
    demText += utterance.text

print("demText generated!")

# write that string into a text doc 
with open("demTest.txt", "w", encoding="utf-8-sig") as f:
    f.write(demText)

print("demTest written!")

# creates text of r/republicans by going through every utterance in the subreddit and adding it with a newline to a string
repText = ""
repGenerator = repCorpus.iter_utterances()
for utterance in repGenerator:
    repText += "\n"
    repText += utterance.text

print("repText generated!")

# write that string into a text doc 
with open("repTest.txt", "w", encoding="utf-8-sig") as f:
    f.write(repText)

print("repTest written!")


def get_sentences(text: str) -> list[list[str]]:
    """Split the specified text into sentences, consisting of text tokens."""

    sents = []

    # We process the text in chunks by paragraph, ensuring that a sentence
    # never crosses a paragraph boundary:
    for para in text.split("\n\n"):
        doc = nlp(para.replace("\n", " "))
        for sent in doc.sents:
            tokens = [
                token.text.lower().strip()
                for token in sent
                if not token.is_space
            ]
            sents.append(tokens)

    return sents

#tokenize the subreddits
demSents = get_sentences(demText)
repSents = get_sentences(repText)

print("subreddits tokenized!")

demModel = Word2Vec(demSents, epochs=25, window=4)
repModel = Word2Vec(repSents, epochs=25, window=4)

demVec = demModel.wv
repVec = repModel.wv

def similarity_print(vec, target: str, subName: str, n: int = 10):

    print("Top " + str(n) + " words similar to " + target + " in " + subName + ":")

    for word, value in vec.most_similar(target, topn=n):

        print(f"{value: .2f} {word}")

similarity_print(demVec, 'taxes', 'democrats')
similarity_print(repVec, 'taxes', 'republicans')