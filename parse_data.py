#!/usr/bin/env python3

# import nltk
# nltk.download('punkt')
from convokit import Corpus, download
demCorpus = Corpus(filename=download("subreddit-democrats"))
repCorpus = Corpus(filename=download("subreddit-republicans"))

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


# creates text of r/democrats by going through every utterance in the subreddit and adding it with a newline to a string
demText = ""

demGenerator = demCorpus.iter_utterances()

for utterance in demGenerator:
    demText += "\n"
    demText += utterance.text

print("demText generated!")

# creates text of r/republicans by going through every utterance in the subreddit and adding it with a newline to a string
repText = ""
repGenerator = repCorpus.iter_utterances()
for utterance in repGenerator:
    repText += "\n"
    repText += utterance.text

print("repText generated!")

# tokenize the subreddits
demSents = get_sentences(demText)
repSents = get_sentences(repText)

# write that string into a text doc
with open("corpora/demText.txt", "w", encoding="utf-8-sig") as f:
    f.write(demText)

print("demText written!")

# write that string into a text doc
with open("corpora/repText.txt", "w", encoding="utf-8-sig") as f:
    f.write(repText)

print("repText written!")


print("subreddits tokenized!")
