from gensim.models import Word2Vec

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


def similarity_print(vec, target: str, subName: str, n: int = 10):

    print("Top " + str(n) + " words similar to " + target + " in " + subName + ":")

    for word, value in vec.most_similar(target, topn=n):

        print(f"{value: .2f} {word}")


similarity_print(demVec, "taxes", "democrats")
similarity_print(repVec, "taxes", "republicans")
