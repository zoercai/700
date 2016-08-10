#!/usr/bin/env python

from urllib.request import urlopen
import json
import codecs

api_key = 'aaa8ed1a-d2e0-42f4-9437-11e77e48244b'

# url = 'http://content.guardianapis.com/search?order-by=oldest&q=olympics&api-key=%s' % api_key
url = 'http://content.guardianapis.com/search?order-by=oldest&q=olympics&api-key=aaa8ed1a-d2e0-42f4-9437-11e77e48244b'
json_obj = urlopen(url)
reader = codecs.getreader("utf-8")

data = json.load(reader(json_obj))

for item in data['response']['results']:
    print(item)
