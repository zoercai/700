import logging
import sys
import warnings
import numpy as np
import os
from Cluster.Node import Node
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import StanfordNERTagger
from sklearn import metrics
from sklearn import decomposition
from sklearn.pipeline import make_pipeline
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import euclidean_distances
from Cluster.Link import Link
from os.path import dirname, abspath
# import matplotlib.pyplot as plt

d = dirname(dirname(abspath(__file__)))
nltk.data.path.append(str(d)+'/static/')


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')  # Reads all words and drops everything else
    tokens = tokenizer.tokenize(text)

    filtered_tokens = [word for word in tokens if (word not in stopwords.words('english'))]  # Filters out stopwords

    # # Using NE only - not recommended
    # parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # ne_tagger = os.path.join(parent_folder, 'stanford-ner.jar')
    # ne_type = os.path.join(parent_folder, 'english.conll.4class.distsim.crf.ser.gz')
    # st = StanfordNERTagger(ne_type, ne_tagger)
    # tokens = st.tag(filtered_tokens)
    # ne = [token for token, tag in tokens if tag != 'O']
    # logging.debug(ne)
    # return ne

    # Turns words into their bases
    lemmatizer = WordNetLemmatizer()
    post_to_lem = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
    post_to_lem = {'NN': 'n'}
    lemmatized_tokens = [lemmatizer.lemmatize(i, post_to_lem[j[:2]]) for i, j in pos_tag(filtered_tokens) if j[:2] in post_to_lem]
    # logging.debug(lemmatized_tokens)

    return lemmatized_tokens


def cluster(articles_list, no_of_clusters):
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # to remove warnings from k-means method
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    # Add articles into dictionary
    token_dict = {}
    article_content_dict = {}
    for article in articles_list:
        token_dict[article.name] = article.body
        article_content_dict[article.name] = article.bodyhtml
    # for headline in token_dict.keys():
    #     logging.debug(headline)

    # Convert the tokens into matrix of tfidf values
    max_features = no_of_clusters * 4

    articles_content = [article.body for article in articles_list]

    # Calculating tf-df
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features=max_features, lowercase=False)
    tfidf_matrix = tfidf_vectorizer.fit_transform(articles_content)

    # Using only term frequency
    # tf_vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english', max_features=None, lowercase=False)
    # tfidf_matrix = tf_vectorizer.fit_transform(token_dict.values())

    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names()
    # feature_names = tf_vectorizer.get_feature_names()
    logging.debug(feature_names)

    # # Convert using hashing vectorizer instead - not recommended
    # hasher = HashingVectorizer(n_features=max_features,
    #                            stop_words='english', non_negative=True,
    #                            norm=None, binary=False)
    # hasing_vectorizer = make_pipeline(hasher, TfidfTransformer())
    # tfidf_matrix = hasing_vectorizer.fit_transform(token_dict.values())

    final_matrix = tfidf_matrix.todense()
    logging.debug("Document points positions:")
    logging.debug(final_matrix)

    # Hierarchical clustering
    def hierarchical(no_of_clusters):
        # for linkage in ('ward', 'average', 'complete'):
        linkage = 'complete'
        h_clustering = AgglomerativeClustering(linkage=linkage, n_clusters=no_of_clusters)
        h_clusters = h_clustering.fit_predict(final_matrix)
        logging.debug("Article clusters, method: " + linkage)
        logging.debug(h_clusters)
        h_cluster_centers = []
        for i in range(0, no_of_clusters):
            article_indices = [j for j, cluster_num in enumerate(h_clusters) if cluster_num == i]
            logging.debug(article_indices)
            cluster_articles = final_matrix[article_indices, :]
            # logging.debug(cluster_articles)
            centroids_coord = cluster_articles.mean(axis=0)
            h_cluster_centers.append(list(np.array(centroids_coord).reshape(-1,)))
        h_cluster_centers = np.array(h_cluster_centers)
        silhouette_score = metrics.silhouette_score(final_matrix, h_clustering.labels_)
        logging.debug("Silhouette score for %d clusters: " % no_of_clusters)
        logging.debug(silhouette_score)
        return silhouette_score, h_clustering, h_clusters, h_cluster_centers

    h_silhouette_scores = [0.0, 0.0]
    for j in range(2, no_of_clusters+1):
        silhouette_score, h_clustering, h_clusters, h_cluster_centers = hierarchical(j)
        h_silhouette_scores.append(silhouette_score)
    # Get index of max
    best_cluster_number = h_silhouette_scores.index(max(h_silhouette_scores))
    silhouette_score, h_clustering, h_clusters, h_cluster_centers = hierarchical(best_cluster_number)
    h_silhouette_scores.append(silhouette_score)
    logging.debug("Final silhouette score for %d clusters: " % best_cluster_number)
    h_final_silhouette = metrics.silhouette_score(final_matrix, h_clustering.labels_)
    logging.debug(h_final_silhouette)

    # Mini-batch k-means clustering
    x_silhouette_scores = [0.0, 0.0]
    for i in range(2, no_of_clusters+1):
        k_clustering = MiniBatchKMeans(n_clusters=i, init='k-means++', n_init=1, verbose=0)
        k_clusters = k_clustering.fit_predict(final_matrix)
        logging.debug("Article clusters, method: k-means")
        logging.debug(k_clusters)
        logging.debug("silhouette_score for %d clusters: " % i)
        silhouette_score = metrics.silhouette_score(final_matrix, k_clustering.labels_)
        logging.debug(silhouette_score)
        x_silhouette_scores.append(silhouette_score)
    # Get index of max (k-means)
    best_cluster_number = x_silhouette_scores.index(max(x_silhouette_scores))
    x_clustering = MiniBatchKMeans(n_clusters=best_cluster_number, init='k-means++', n_init=1, verbose=0)
    x_clusters = x_clustering.fit_predict(final_matrix)
    logging.debug("Final silhouette score for %d clusters: " % best_cluster_number)
    x_final_silhouette = metrics.silhouette_score(final_matrix, x_clustering.labels_)
    logging.debug(x_final_silhouette)

    # # DBSCAN clustering — used for checking data concentration
    # db_clustering = DBSCAN(eps=0.00001, min_samples=1)
    # db_clusters = clustering.fit_predict(final_matrix)
    # # print(metrics.silhouette_score(final_matrix, clustering.labels_))
    # logging.debug("Article clusters, method: DBSCAN")
    # logging.debug(db_clusters)

    # Set clustering algorithm
    clustering = h_clustering
    clusters = h_clusters
    cluster_centers = h_cluster_centers
    if x_final_silhouette > h_final_silhouette:
        clustering = x_clustering
        clusters = x_clusters
        cluster_centers = x_clustering.cluster_centers_
        logging.debug("Winning silhouette score: ")
        logging.debug(x_final_silhouette)
        logging.debug('X-means wins!')
    else:
        logging.debug("Winning silhouette score: ")
        logging.debug(h_final_silhouette)
        logging.debug('Hierachical wins!')

    # # ----------------------------------------------------------------
    # # ----------------------------------------------------------------
    # # Start of local visualisation
    #
    # # Reduce dimensionality to 2 for plotting
    # # PCA
    # pca = decomposition.PCA(n_components=2)
    # reduced_matrix = pca.fit_transform(final_matrix)
    #
    # # # LSA
    # # svd = TruncatedSVD(2)
    # # normalizer = Normalizer(copy=False)
    # # lsa = make_pipeline(svd, normalizer)
    # # reduced_matrix = lsa.fit_transform(final_matrix)
    #
    # # Visualize the clustering
    # def plot_clustering(reduced_matrix, labels, title=None):
    #     plt.figure()
    #     x_min = min(point[0] for point in reduced_matrix) - 0.2
    #     x_max = max(point[0] for point in reduced_matrix) + 0.2
    #     y_min = min(point[1] for point in reduced_matrix) - 0.2
    #     y_max = max(point[1] for point in reduced_matrix) + 0.2
    #     plt.axis([x_min, x_max, y_min, y_max])
    #     counter = 1
    #     for i in range(reduced_matrix.shape[0]):
    #         plt.text(reduced_matrix[i, 0], reduced_matrix[i, 1], counter,
    #                  color=plt.cm.spectral(labels[i]/10.),
    #                  fontdict={'weight': 'bold', 'size': 12})
    #         counter += 1
    #     if title is not None:
    #         plt.title(title, size=17)
    #     plt.axis('on')
    #     plt.tight_layout()
    #
    # print(clustering.labels_)
    # print(reduced_matrix)
    # plot_clustering(reduced_matrix, clustering.labels_)
    # plt.show()
    #
    # # Plot the points
    # count = 1
    # for f1, f2 in reduced_matrix:
    #     plt.scatter(f1, f2)
    #     plt.annotate(count, (f1, f2))
    #     count += 1
    # plt.show()
    # # ----------------------------------------------------------------
    # # ----------------------------------------------------------------

    return final_matrix, tfidf_vectorizer, clusters, cluster_centers
