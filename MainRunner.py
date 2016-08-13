from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from ArticlesRetriever import retrieve_articles
from Clusterer import cluster
import json
import jsonpickle

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cluster')
def clusterer():
    start_date = request.args.get('start_date', 0, type=int)
    end_date = request.args.get('end_date', 0, type=int)

    # retrieve articles
    articles_list = retrieve_articles(start_date, end_date)

    # process articles
    node_list, link_list = cluster(articles_list)

    # format & jsonify
    # nodes_and_links = {'nodes': node_list, 'links': link_list}
    # final = jsonpickle.encode(nodes_and_links)
    # print(final)
    json_nodelist = json.dumps([ob.__dict__ for ob in node_list])
    json_linklist = json.dumps([ob.__dict__ for ob in link_list])
    final = '{"nodes":' + json_nodelist + ', "links":' + json_linklist + '}'

    return final


if __name__ == '__main__':
    app.run()
