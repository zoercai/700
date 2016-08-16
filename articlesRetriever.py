# Given the date range of the articles to retrieve, retrieves article plain body text from the Guardian API and returns
#  a list of Article objects.

import codecs
import json
from urllib.request import urlopen
from Article import Article


def retrieve_articles(results, from_date, to_date):
    api_key = 'aaa8ed1a-d2e0-42f4-9437-11e77e48244b'

    # results = 50
    query = ''   # Optional, only used for testing

    if results == '':
        results = '50'
    if from_date == '':
        from_date = '2016-07-10'
    if to_date == '':
        to_date = '2016-08-10'

    url = 'http://content.guardianapis.com/search?q=' + query
    url += '&show-blocks=body'
    url += '&page-size=' + str(results)
    url += '&from-date=' + str(from_date)
    url += '&to-date=' + str(to_date)
    url += '&api-key=' + api_key
    print(url)

    json_obj = urlopen(url)
    reader = codecs.getreader("utf-8")

    data = json.load(reader(json_obj))

    articles = []

    for item in data['response']['results']:
        name = item['webTitle']
        url = item['webUrl']
        body = item['blocks']['body']
        if len(body) > 0:
            new_article = Article(name, url, body[0]['bodyTextSummary'])
            articles.append(new_article)

    return articles
