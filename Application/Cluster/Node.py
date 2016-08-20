class Node:
    id = ''
    group = ''
    features = ''
    bodyhtml = ''
    silhouette = ''

    def __init__(self, id, group, features, bodyhtml, silhouette="1"):
        self.id = id
        self.group = group
        self.features = features
        self.bodyhtml = bodyhtml
        self.silhouette = silhouette

    # def __init__(self, id, group, features, bodyhtml):
    #     self.id = id
    #     self.group = group
    #     self.features = features
    #     self.bodyhtml = bodyhtml
