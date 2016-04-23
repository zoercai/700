import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


class Preprocessor:
    def tokenize(self, file):
        content = file.read()
        tokenizer = RegexpTokenizer(r'\w+')

        tokens = tokenizer.tokenize(content)
        tokens = [token.lower() for token in tokens]
        tagged = nltk.pos_tag(tokens)

        filtered_words = [word for word in tokens if word not in stopwords.words('english')]

        print(tokens)
        print(len(tokens))
        print(filtered_words)
        print(len(filtered_words))
        # print(re.sub('[^a-zA-Z0-9 _-]','',line).lower().split())

#
#
# print("hello world")
#
# name = "Zoe wants a burger, haha."
#
# word_list = name.split(" ")
#
# print(word_list)
#
# test_file = open("test.txt")
#
# print(test_file.read())
#
# test_file.close()
#
#
#
# class Document:
#     __name = None
#     __totalWords = 0
#
#     def __init__(self, name, words):
#         self.__name = name;
#
#
#     def set_name(self, name):
#         self.__name = name;
#
#     def get_name(self):
#         return self.__name
#
#     def get_type(self):
#         print("Document")
#
#
# cat = Document("hello")
#
#
# class SubDocument(Document):
#
#     def __init__(self, name, words, blah):
#         super(SubDocument, self).__init__(name, words)
#
#     def overloading(self, optionValue=None):
#         if optionValue is None:
#             # do nothing
#         else:
#             # do something
#
