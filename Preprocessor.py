import re
import nltk
from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


class Preprocessor:
    def tokenize(self, file):
        content = file.read()

        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(content)
        tokens = [token.lower() for token in tokens]

        filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else lemmatizer.lemmatize(i) for i, j in
              pos_tag(filtered_tokens)]

        # tagged = nltk.pos_tag(lemmatized_tokens)

        print(tokens)
        print(len(tokens))
        print(filtered_tokens)
        print(len(filtered_tokens))
        print(lemmatized_tokens)
        print(len(lemmatized_tokens))
        # print(tagged)
        return lemmatized_tokens



    def mapTerm(self, tokens):
        termMap = {}

        for token in tokens:
            if token in termMap:
                termMap[token] = termMap[token] + 1
            else:
                termMap[token] = 1

        print(termMap.items())
        return termMap

