#!/usr/bin/env python3

import nltk
nltk.download('punkt')
from convokit import Corpus, download

with open("subreddit_names.txt") as f:
    for line in f:
        name = line.strip()
        corpus = Corpus(filename=download(f"subreddit-{name}"))
        print(f"{name} downloaded!")

        text = ""
        generator = corpus.iter_utterances()
        for utterance in generator:
            text += "\n"
            text += utterance.text
        print(f"{name} text generated!")

        with open(f"corpora/{name}.txt", "w", encoding="utf-8-sig") as f:
            f.write(text)
        print(f"{name} text written!")