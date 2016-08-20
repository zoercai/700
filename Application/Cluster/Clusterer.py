from __future__ import print_function


import logging
import sys
import warnings
import numpy as np

from Link import Link
from Node import Node
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')  # Reads all words and drops everything else
    tokens = tokenizer.tokenize(text)

    filtered_tokens = [word for word in tokens if (word not in stopwords.words('english'))]  # Filters out stopwords

    # parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # tagger = os.path.join(parent_folder, 'stanford-ner.jar')
    # type = os.path.join(parent_folder, 'english.conll.4class.distsim.crf.ser.gz')
    # st = StanfordNERTagger(type, tagger)
    # tokens = st.tag(filtered_tokens)
    # ne = [token for token, tag in tokens if tag != 'O']
    # # print(ne)
    # return ne

    # Turns words into their bases
    lemmatizer = WordNetLemmatizer()
    post_to_lem = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
    post_to_lem = {'NN': 'n'}
    lemmatized_tokens = [lemmatizer.lemmatize(i, post_to_lem[j[:2]]) for i, j in pos_tag(filtered_tokens) if j[:2] in post_to_lem]
    # lemmatized_tokens = [lemmatizer.lemmatize(i, post_to_lem[j[:3]]) for i, j in pos_tag(filtered_tokens) if j[:3] in post_to_lem]
    # logging.debug(lemmatized_tokens)

    return lemmatized_tokens


def cluster(articles_list, no_of_clusters):
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # to remove warnings from k-means method
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    # Add articles into dictionary
    token_dict = {}
    article_content_dict = {}
    for article in articles_list:
        token_dict[article.name] = article.body
        article_content_dict[article.name] = article.bodyhtml
    for headline in token_dict.keys():
        logging.info(headline)

    # Convert the tokens into matrix of tfidf values
    max_features = no_of_clusters * 4

    articles_content = [article.body for article in articles_list]

    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features=max_features, lowercase=False)
    tfidf_matrix = tfidf_vectorizer.fit_transform(articles_content)

    # tf_vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english', max_features=None, lowercase=False)
    # tfidf_matrix = tf_vectorizer.fit_transform(token_dict.values())

    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names()
    # feature_names = tf_vectorizer.get_feature_names()
    logging.info(feature_names)

    # # Convert using hashing vectorizer instead
    # hasher = HashingVectorizer(n_features=max_features,
    #                            stop_words='english', non_negative=True,
    #                            norm=None, binary=False)
    # hasing_vectorizer = make_pipeline(hasher, TfidfTransformer())
    # tfidf_matrix = hasing_vectorizer.fit_transform(token_dict.values())

    final_matrix = tfidf_matrix.todense()

    logging.info("Document points positions:")
    logging.info(final_matrix)

    k_clusters = no_of_clusters

    # # hierarchical clustering
    # # for linkage in ('ward', 'average', 'complete'):
    # linkage = 'ward'
    # clustering = AgglomerativeClustering(linkage=linkage, n_clusters=k_clusters)
    # clusters = clustering.fit_predict(final_matrix)
    # logging.info("Article clusters, method: " + linkage)
    # logging.info(clusters)

    # x-means clustering
    silhouette_scores = [0.0, 0.0]
    for i in range(2, no_of_clusters+1):
        clustering = MiniBatchKMeans(n_clusters=i, init='k-means++', n_init=1, verbose=0)
        clusters = clustering.fit_predict(final_matrix)
        logging.info("Article clusters, method: k-means")
        # logging.info(clusters)
        logging.info("silhouette_score for %d clusters: " % i)
        silhouette_score = metrics.silhouette_score(final_matrix, clustering.labels_)
        logging.info(silhouette_score)
        silhouette_scores.append(silhouette_score)
    # Get index of max (k-means)
    best_cluster_number = silhouette_scores.index(max(silhouette_scores))
    clustering = MiniBatchKMeans(n_clusters=best_cluster_number, init='k-means++', n_init=1, verbose=0)
    clusters = clustering.fit_predict(final_matrix)
    logging.info("Final silhouette score for %d clusters: " % best_cluster_number)
    logging.info(metrics.silhouette_score(final_matrix, clustering.labels_))

    # # DBSCAN clustering
    # clustering = DBSCAN(eps=0.0000000000000000000000001, min_samples=2)
    # clusters = clustering.fit_predict(final_matrix)
    # # print(metrics.silhouette_score(final_matrix, clustering.labels_))
    # # logging.info("Article clusters, method: k-means")
    # # logging.info(clusters)

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Start of web app processing

    # Turn articles and centroids into nodes
    node_list = []
    final_list = final_matrix.tolist()
    for i, item in enumerate(articles_content):
        features = zip(feature_names, final_list[i])
        sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
        new_article_node = Node(articles_list[i].name, int(clusters[i]), ",".join("(%s,%s)" % tup for tup in sorted_features), articles_list[i].bodyhtml)
        node_list.append(new_article_node)

    for i, centroid_vector in enumerate(clustering.cluster_centers_):
        order_centroids = clustering.cluster_centers_.argsort()[:, ::-1]
        top_features = []
        for ind in order_centroids[i, :10]:
            top_features.append(str(feature_names[ind]) + ": " + str(clustering.cluster_centers_[i, ind]))
        new_centroid_node = Node("centroid_" + str(i), int(i), str(top_features), str(top_features))
        node_list.append(new_centroid_node)

    # Append main centroid
    main_centroid = Node("centroid_main", k_clusters, tfidf_vectorizer.get_feature_names(), 'centroid')
    node_list.append(main_centroid)

    # Calculate distances
    def distance_normaliser(distance):
        return int(distance * 10) + 1

    link_list = []

    centroid = np.mean(clustering.cluster_centers_, axis=0)
    inter_centroid_distance_matrix = euclidean_distances(clustering.cluster_centers_, centroid)

    logging.info("inter-centroid distances")
    logging.info(inter_centroid_distance_matrix)
    for i, row in enumerate(inter_centroid_distance_matrix):
        new_link = Link("centroid_main", "centroid_" + str(i), distance_normaliser(row[0]))
        link_list.append(new_link)

    intra_centroid_distance_matrix = euclidean_distances(final_matrix, clustering.cluster_centers_)
    logging.info("Centroid vectors")
    logging.info(clustering.cluster_centers_)

    print(clusters)
    print([article.name for article in articles_list], sep='\n')
    for i, row in enumerate(intra_centroid_distance_matrix):
        centroid_num = clusters[i]
        distance = distance_normaliser(row[centroid_num])
        new_link = Link(articles_list[i].name, "centroid_" + str(centroid_num), distance)
        link_list.append(new_link)

    # ----------------------------------------------------------------

    # # Do not uncomment this unless you want to see a mess
    # for i, row in enumerate(intra_centroid_distance_matrix):
    #     for j, distance in enumerate(row):
    #         new_link = Link(articles_list[i].name, "centroid_" + str(j), distance)
    #         link_list.append(new_link)

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------

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
    # show()
    #
    # # Plot the points
    # count = 1
    # for f1, f2 in reduced_matrix:
    #     plt.scatter(f1, f2)
    #     plt.annotate(count, (f1, f2))
    #     count += 1
    # show()

    return node_list, link_list
