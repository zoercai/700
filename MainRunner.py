from Preprocessor import Preprocessor
import sys

file = open("test.txt")

preprocessor = Preprocessor()
preprocessor.tokenize(file)


