<<<<<<< HEAD
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
=======
import json
from flask import Flask, render_template, request, Response
from ArticlesRetriever import retrieve_articles
from Cluster.Clusterer import cluster

app = Flask(__name__)
>>>>>>> master


@app.route('/')
def index():
    return render_template('index.html')

<<<<<<< HEAD


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens]
=======
>>>>>>> master

@app.route('/cluster')
def clusterer():
    results = request.args.get('results')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    clusters = request.args.get('clusters', 20, type=int)

    # retrieve articles
    articles_list = retrieve_articles(results, start_date, end_date)

    # process articles
    node_list, link_list = cluster(articles_list, clusters)

    # format & jsonify
    json_nodelist = json.dumps([ob.__dict__ for ob in node_list])
    json_linklist = json.dumps([ob.__dict__ for ob in link_list])
    final = '{"nodes":' + json_nodelist + ', "links":' + json_linklist + '}'

    # print(final)

    resp = Response(response=final,
                    status=200,
                    mimetype="application/json")

    return resp


<<<<<<< HEAD
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




=======
if __name__ == '__main__':
    app.run()
>>>>>>> master
