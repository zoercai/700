
class Article:
    name = ''
    url = ''
    body = ''
    distance = 0

    def __init__(self, name, url, body):
        self.name = name
        self.url = url
        self.body = body

    def distance(self, distance):
        self.distance = distance