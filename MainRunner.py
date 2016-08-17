import string
import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

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
number_of_features = 20;


# 834 - 20 - 1: 833
# 813 : 833
# ordered_matrix[0:, (813:833)]
# Currently grabbing the last 20 of the ordered_matrix (Most highest 20 values)
top_feature_matrix = ordered_matrix[0:, (ncol-number_of_features-1):(ncol-1)]


# print(top_feature_matrix)
# print(tfidf_matrix.shape)

# Get tokens values (TODO not correct after extracting top features)
# feature_names = tfidf_vectorizer.get_feature_names()
# print(feature_names)


# Calculate centroid - (the mean of the top 20). Takes each word from top_feature_matrix and calculates the average value
# regarding each document
centroid = numpy.mean(top_feature_matrix, axis=0)
print("Centroid: ")
print(centroid)
print("Centroid similarities: ")

# Comparing each document in top_feature_matrix with the centroid
centroid_similarities = cosine_similarity(centroid, top_feature_matrix)
print(centroid_similarities)

# Calculate average similarity
mean_similarity = numpy.mean(centroid_similarities)
print("MEAN SIMILARITY: ")
print(mean_similarity)
print("//")



# PCA decomposes the 20 dimension vector into a 2 dimension vector
pca = decomposition.PCA(n_components=2)
top_feature_matrix = pca.fit_transform(top_feature_matrix)
print("Decomposed Matrix: ")
print(top_feature_matrix)

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(top_feature_matrix)
t_batch = time.time() - t0

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = numpy.unique(k_means_labels)


fig = plt2.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

print("SILHOUETTE VALUES:")
print(silhouette_samples(top_feature_matrix, k_means_labels))

ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(3), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(top_feature_matrix[my_members, 0], top_feature_matrix[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt2.show()




