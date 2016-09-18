class Node:
    # Represents a node (either an article or a centroid) in the web app visualisation

    id = ''
    group = ''
    features = ''
    bodyhtml = ''

    def __init__(self, id, group, features, bodyhtml):
        self.id = id
        self.group = group
        self.features = features
        self.bodyhtml = bodyhtml
