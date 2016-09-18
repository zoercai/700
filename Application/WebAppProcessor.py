from sklearn.metrics import euclidean_distances
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from Cluster.Link import Link
from Cluster.Node import Node


def process(final_matrix, tfidf_vectorizer, articles_list, clusters, cluster_centers):
    # Turns clustered articles and their centroids into nodes for web application visualisation

    feature_names = tfidf_vectorizer.get_feature_names()
    articles_content = [article.body for article in articles_list]
    no_of_clusters = len(cluster_centers)

    # Turn articles and centroids into nodes
    node_list = []
    final_list = final_matrix.tolist()
    for i, item in enumerate(articles_content):
        features = zip(feature_names, final_list[i])
        sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
        new_article_node = Node(articles_list[i].name, int(clusters[i]),
                                ",".join("(%s,%s)" % tup for tup in sorted_features), articles_list[i].bodyhtml)
        node_list.append(new_article_node)

    for i, centroid_vector in enumerate(cluster_centers):
        order_centroids = cluster_centers.argsort()[:, ::-1]
        top_features = []
        for ind in order_centroids[i, :10]:
            top_features.append(str(feature_names[ind]) + ": " + str(cluster_centers[i, ind]))
        new_centroid_node = Node("centroid_" + str(i), int(i), str(top_features), str(top_features))
        node_list.append(new_centroid_node)

    # Append main centroid
    main_centroid = Node("centroid_main", no_of_clusters, feature_names, 'centroid')
    node_list.append(main_centroid)

    # Calculate distances
    def distance_normaliser(distance):
        return int(distance * 10) + 1

    link_list = []

    centroid = np.mean(cluster_centers, axis=0)
    inter_centroid_distance_matrix = euclidean_distances(cluster_centers, centroid)

    logging.debug("inter-centroid distances")
    logging.debug(inter_centroid_distance_matrix)
    for i, row in enumerate(inter_centroid_distance_matrix):
        new_link = Link("centroid_main", "centroid_" + str(i), distance_normaliser(row[0]))
        link_list.append(new_link)

    intra_centroid_distance_matrix = euclidean_distances(final_matrix, cluster_centers)
    logging.debug("Centroid vectors")
    logging.debug(cluster_centers)

    logging.debug("Final clusters")
    logging.debug(clusters)
    logging.debug("Article titles")
    logging.debug([article.name for article in articles_list])
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

    return node_list, link_list
