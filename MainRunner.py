from Preprocessor import Preprocessor
import sys

file = open("test.txt")

preprocessor = Preprocessor()
tokens = preprocessor.tokenize(file)
map = preprocessor.mapTerm(tokens)


