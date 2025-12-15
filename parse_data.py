#!/usr/bin/env python3

import nltk
nltk.download('punkt')
from convokit import Corpus, download

with open("subreddit_names.txt") as f:  
    for line in f:
        name = line.strip() # for each subreddit name in the subredditnames.txt file:

        # download the subreddit from convokit
        # end index of 500K added to keep download managable
        # we only need the text data from the utterances so we can exclude the other metadeta
        corpus = Corpus(filename=download(f"subreddit-{name}"), utterance_end_index=500000, exclude_utterance_meta=["score", "top_level_comment", "retrieved_on", "gilded", "gildings", "stickied", "permalink", "author_flair_text"])
        print(f"{name} downloaded!")

        # create a string containing the text of each utterance each on a newline
        text = ""
        generator = corpus.iter_utterances()
        for utterance in generator:
            text += f"{utterance.text}\n"
        print(f"{name} text generated!")

        # write out that string to a text doc named after the subreddit
        with open(f"corpora/{name}.txt", "w", encoding="utf-8-sig") as f:
            f.write(text)
        print(f"{name} text written!")