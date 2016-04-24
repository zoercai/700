import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy

from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens]

    filtered_tokens = [word for word in tokens if ((word not in stopwords.words('english')))]

    # print(pos_tag(filtered_tokens))

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else lemmatizer.lemmatize(i) for i, j
        in
        pos_tag(filtered_tokens)]
    # print(filtered_tokens)
    return lemmatized_tokens


token_dict = {}

# Read in all test files
for subdir, dirs, files in os.walk(os.getcwd()+"/tests1"):
    for file in files:
        if file.endswith(".txt"):
            file_path = subdir + os.path.sep + file
            document = open(file_path, 'r')
            text = document.read()
            token_dict[file] = text

print(token_dict.keys())

# Create tokenizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
# tfidf_vectorizer = TfidfVectorizer()

# Convert the tokens into matrix of tfidf values
tfidf_matrix = tfidf_vectorizer.fit_transform(token_dict.values())
# print(token_dict.values())
print(tfidf_matrix.shape)

# Get tokens values
feature_names = tfidf_vectorizer.get_feature_names()
# print(feature_names)

# Calculate centroid
centroid = numpy.mean(tfidf_matrix.todense(), axis=0)
print("Centroid: ")
print(centroid)
print("Centroid similarities: ")
centroid_similarities = cosine_similarity(centroid, tfidf_matrix)
print(centroid_similarities)

# Calculate average similarity
mean_similarity = numpy.mean(centroid_similarities)
print(mean_similarity)

for similarity in centroid_similarities[0]:
    if similarity > mean_similarity:
        print(similarity)

# similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
# print("Similarities to first: ")
# print(similarities)

