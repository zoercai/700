import os
import numpy
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
for subdir, dirs, files in os.walk(os.getcwd()+"/tests3"):
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
number_of_features = 5
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
top_feature_matrix_pca = pca.fit_transform(top_feature_matrix)
print("top features:")
print(top_feature_matrix_pca)


count = 1;
for f1, f2 in top_feature_matrix_pca:
    plt.scatter( f1, f2 )
    plt.annotate(count, (f1, f2))
    count=count+1

# plt.show()

# Calculate cosine similarity
similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Similarities to first: ")
print(similarities)


from time import time
import numpy as np
from matplotlib import pyplot as plt

#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure()
    count = 1
    for i in range(X_red.shape[0]):

        plt.text(X_red[i, 0], X_red[i, 1], count,
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
        count=count+1

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('on')
    plt.tight_layout()

#----------------------------------------------------------------------
print("Computing embedding")
X_red = top_feature_matrix_pca
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=3)
    t0 = time()
    clustering.fit(X_red)
    print("%s : %.2fs" % (linkage, time() - t0))

    # plt.figure()
    # count = 1
    # for f1, f2 in X_red:
    #     plt.scatter(f1, f2)
    #     plt.annotate(count, (f1, f2))
    #     count = count + 1

    plot_clustering(X_red, top_feature_matrix_pca, clustering.labels_, "%s linkage" % linkage)


plt.show()

