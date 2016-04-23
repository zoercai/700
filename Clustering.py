from sklearn.feature_extraction.text import TfidfVectorizer

class Clustering:
    # def __init__(self):
    #     iris = datasets.load_iris()
    #     digits = datasets.load_digits()
    #     print(digits)

    def tfidf(self, tokens):
        tfidfer = TfidfVectorizer()
        tfs = tfidfer.fit_transform(tokens)
        # print(tfidfer)
        # print(tfs.toarray())

        feature_names = tfidfer.get_feature_names()
        for col in tfs.nonzero()[1]:
            print(feature_names[col], ' - ', tfs[0, col])