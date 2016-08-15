import logging
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, scatter, annotate
from Cluster.Node import Node
from Cluster.Link import Link
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn import decomposition
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')  # Reads all words and drops everything else
    tokens = tokenizer.tokenize(text)

    filtered_tokens = [word for word in tokens if (word not in stopwords.words('english'))]  # Filters out stopwords

    # Turns words into their bases
    lemmatizer = WordNetLemmatizer()
    post_to_lem = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
    lemmatized_tokens = [lemmatizer.lemmatize(i, post_to_lem[j[:2]])
                         for i, j in pos_tag(filtered_tokens) if j[:2] in post_to_lem]
    logging.debug(lemmatized_tokens)
    return lemmatized_tokens


def cluster(articles_list):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    # Add articles into dictionary
    token_dict = {}
    for article in articles_list:
        token_dict[article.name] = article.body
    for headline in token_dict.keys():
        logging.info(headline)

    # Convert the tokens into matrix of tfidf values
    max_features = 10
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(token_dict.values())

    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names()
    logging.info(feature_names)

    # Reduce dimensionality to 2 for plotting
    pca = decomposition.PCA(n_components=2)
    reduced_matrix = pca.fit_transform(tfidf_matrix.todense())

    logging.info("Document points positions:")
    logging.info(reduced_matrix)

    k_clusters = 5

    # # hierarchical clustering
    # for linkage in ('ward', 'average', 'complete'):
    #     clustering = AgglomerativeClustering(linkage=linkage, n_clusters=k_clusters)
    #     print(clustering.fit_predict(reduced_matrix))

    # k-means clustering
    clustering = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', n_init=1, verbose=0)
    # print(clustering.fit_transform(reduced_matrix))
    clusters = clustering.fit_predict(reduced_matrix)
    print(clusters)

    # Turn articles and centroids into nodes
    node_list = []
    for i, item in enumerate(articles_list):
        new_article_node = Node(item.name, int(clusters[i]))
        node_list.append(new_article_node)

    for i in range(0, k_clusters):
        new_centroid_node = Node("centroid_" + str(i), int(i))
        node_list.append(new_centroid_node)

    link_list = []
    distance_matrix = euclidean_distances(reduced_matrix, clustering.cluster_centers_)
    for i, row in enumerate(distance_matrix):
        centroid_num = clusters[i]
        distance = int(row[centroid_num]*10) + 1
        new_link = Link(articles_list[i].name, "centroid_" + str(centroid_num), distance)
        link_list.append(new_link)

    # for i, row in enumerate(distance_matrix):
    #     for j, distance in enumerate(row):
    #         new_link = Link(articles_list[i].name, "centroid_" + str(j), distance)
    #         link_list.append(new_link)

    # Plot the points
    count = 1
    for f1, f2 in reduced_matrix:
        plt.scatter(f1, f2)
        plt.annotate(count, (f1, f2))
        count += 1
    show()

    return node_list, link_list
