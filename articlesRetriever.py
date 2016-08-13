# Given the date range of the articles to retrieve, retrieves article plain body text from the Guardian API and returns
#  a list of Article objects.

from urllib.request import urlopen
import json
import codecs

api_key = 'aaa8ed1a-d2e0-42f4-9437-11e77e48244b'

query = 'olympics'   # Optional, only used for testing
fromDate = '2016-07-10'
toDate = '2016-08-10'

url = 'http://content.guardianapis.com/search?q=' + query
url += '&show-blocks=body'
url += '&from-date=' + fromDate
url += '&to-date=' + toDate
url += '&api-key=' + api_key
print(url)

json_obj = urlopen(url)
reader = codecs.getreader("utf-8")

data = json.load(reader(json_obj))

for item in data['response']['results']:
    print(item['blocks']['body'][0]['bodyTextSummary'])
    print('New article \n')
