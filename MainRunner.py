import string
import os
import numpy
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import cluster
from sklearn import decomposition

from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens]

    filtered_tokens = [word for word in tokens if ((word not in stopwords.words('english')))]

    # for word, tag in pos_tag(filtered_tokens):
    #     if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
    #         print(word)
    #         print(tag)

    filtered_tokens = [word for word, tag in pos_tag(filtered_tokens) if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS']
    # print(filtered_tokens)

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

# Order the matrix from least sum to most
ordered_matrix = numpy.take(tfidf_matrix.todense(), numpy.sum(tfidf_matrix.todense(), axis=0).argsort(),axis=1)

# Extract the highest sum features (columns)
ncol = ordered_matrix.shape[1]
number_of_features = 20
top_feature_matrix = ordered_matrix[0:, (ncol-number_of_features-1):(ncol-1)]
print(top_feature_matrix)
print(tfidf_matrix.shape)

# Get tokens values (TODO not correct after extracting top features)
# feature_names = tfidf_vectorizer.get_feature_names()
# print(feature_names)

# Calculate centroid
centroid = numpy.mean(top_feature_matrix, axis=0)
print("Centroid: ")
print(centroid)
print("Centroid similarities: ")
centroid_similarities = cosine_similarity(centroid, top_feature_matrix)
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

top_feature_matrix = numpy.concatenate((top_feature_matrix,centroid),axis=0)
pca = decomposition.PCA(n_components=2)
top_feature_matrix = pca.fit_transform(top_feature_matrix)
print("top features:")
print(top_feature_matrix)


count = 1;
for f1, f2 in top_feature_matrix:
    plt.scatter( f1, f2 )
    plt.annotate(count, (f1, f2))
    count=count+1

plt.show()

# Calculate cosine similarity
similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Similarities to first: ")
print(similarities)

k_means = cluster.KMeans(n_clusters=4)
k_means.fit()

