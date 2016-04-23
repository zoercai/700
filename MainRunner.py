import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens]

    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else lemmatizer.lemmatize(i) for i, j
        in
        pos_tag(filtered_tokens)]
    return lemmatized_tokens


token_dict = {}

# Read in all test files
for subdir, dirs, files in os.walk(os.getcwd()+"/tests"):
    for file in files:
        if file.endswith(".txt"):
            file_path = subdir + os.path.sep + file
            document = open(file_path, 'r')
            text = document.read()
            token_dict[file] = text

# Create tokenizer
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')

# Convert the tokens into matrix of tfidf values
tfs = tfidf.fit_transform(token_dict.values())

# Get tokens values
feature_names = tfidf.get_feature_names()

# Print tfidf's that aren't zero
for col in tfs[0].nonzero()[1]:
    print(feature_names[col], ' - ', tfs[0, col])

# Testing with one single file
testFile = open(os.getcwd()+"/tests/na.6.txt", 'r').read()
response = tfidf.transform([testFile])

# for col in response.nonzero()[1]:
    # print(feature_names[col], ' - ', response[0, col])


