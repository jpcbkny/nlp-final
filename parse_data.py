#!/usr/bin/env python3

# import nltk
# nltk.download('punkt')
from convokit import Corpus, download
demCorpus = Corpus(filename=download("subreddit-democrats"))
repCorpus = Corpus(filename=download("subreddit-republicans"))

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

# write that string into a text doc
with open("corpora/demText.txt", "w", encoding="utf-8-sig") as f:
    f.write(demText)

print("demText written!")

# write that string into a text doc
with open("corpora/repText.txt", "w", encoding="utf-8-sig") as f:
    f.write(repText)

print("repText written!")
