class Link:
    # Represents a link (connection) between a node and a centroid in the web app visualisation

    source = ""
    target = ""
    value = ""

    def __init__(self, source, target, value):
        self.source = source
        self.target = target
        self.value = value
