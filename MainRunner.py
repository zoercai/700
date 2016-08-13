from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from ArticlesRetriever import retrieve_articles

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cluster')
def cluster():
    start_date = request.args.get('start_date', 0, type=int)
    end_date = request.args.get('end_date', 0, type=int)
    # retrieve articles

    # process articles
    return jsonify(result=retrieve_articles(start_date, end_date))


if __name__ == '__main__':
    app.run()
