import os
import sys
import logging
import numpy
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition

from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')  # Reads all words and drops everything else
    tokens = tokenizer.tokenize(text)

    filtered_tokens = [word for word in tokens if (word not in stopwords.words('english'))]  # Filters out stopwords

    # Turns words into their bases
    lemmatizer = WordNetLemmatizer()
    post_to_lem = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
    lemmatized_tokens = [lemmatizer.lemmatize(i, post_to_lem[j[:2]]) for i, j in pos_tag(filtered_tokens) if j[:2] in post_to_lem]
    logging.debug(lemmatized_tokens)
    return lemmatized_tokens


token_dict = {}

# Read in all test files
for subdir, dirs, files in os.walk(os.getcwd()+"/tests3"):
    for file in files:
        if file.endswith(".txt"):
            file_path = subdir + os.path.sep + file
            document = open(file_path, 'r')
            text = document.read()
            token_dict[file] = text

logging.info(token_dict.keys())

# Convert the tokens into matrix of tfidf values
max_features = 5
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features=max_features)
tfidf_matrix = tfidf_vectorizer.fit_transform(token_dict.values())

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names()
logging.info(feature_names)

# Calculate centroid
centroid = numpy.mean(tfidf_matrix.todense(), axis=0)
logging.info("Centroid: ")
logging.info(centroid)
logging.info("Centroid similarities: ")
centroid_similarities = cosine_similarity(centroid, tfidf_matrix)
logging.info(centroid_similarities)

# # Calculate average similarity
# mean_similarity = numpy.mean(centroid_similarities)
# logging.info("Mean similarity")
# logging.info(mean_similarity)
#
# for similarity in centroid_similarities[0]:
#     if similarity > mean_similarity:
#         logging.info(similarity)

# similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
# logging.info("Similarities to first: ")
# logging.info(similarities)

matrix_with_centroid = numpy.concatenate((tfidf_matrix.todense(), centroid), axis=0)

# Reduce dimensionality to 2 for plotting
pca = decomposition.PCA(n_components=2)
reduced_matrix = pca.fit_transform(matrix_with_centroid)

logging.info("Document points positions:")
logging.info(reduced_matrix)

# Plot the points
count = 1
for f1, f2 in reduced_matrix:
    plt.scatter(f1, f2)
    plt.annotate(count, (f1, f2))
    count += 1
plt.show()

# # Calculate cosine similarity
# similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
# print("Similarities to first: ")
# print(similarities)

# ----------------------------------------------------------------------
from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=3)
    print(clustering.fit_predict(reduced_matrix))
