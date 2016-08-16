class Article:
    name = ''
    url = ''
    body = ''
    bodyhtml = ''
    distance = 0

    def __init__(self, name, url, body, bodyhtml):
        self.name = name
        self.url = url
        self.body = body
        self.bodyhtml = bodyhtml

    def distance(self, distance):
        self.distance = distance
