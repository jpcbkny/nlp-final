# nlp-final

Our data was too big to fit onto github.
However, it can be easily and quickly obtained by doing the following in the terminal to run our code:

first install requirements:
`pip install -r requirements.txt`

then run our data parser: 
`python parse_data.py`
Note: the "subreddit_names.txt" file can be changed with (CASE SENSITIVE) subreddit names you wish to parse and run the model on instead each separated by a newline 

At this point you will have text files of the selected subreddits.

Then to compare their embedding spaces, run:
`python embedding.py`

Citations:
Procrustes code obtained from https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8 
get_sentences() function from assignment 4