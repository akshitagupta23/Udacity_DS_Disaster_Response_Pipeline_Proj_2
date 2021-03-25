import json
import plotly
import pandas as pd

import sys
sys.path.append("..")

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Layout, Figure
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import Counter



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:,4:].columns
    category_count = (df.iloc[:,4:] != 0).sum().values
    
    # Top 10 words extarcted using NLP
    word_freq={}
    messages = df['message'].tolist()
    for message in messages:
        tokens = tokenize(message)
        for token in tokens:
            if token not in word_freq.keys():
                word_freq[token] = 1
            else:
                word_freq[token] += 1
    
    word_counts = []
    word_names = []
    word_dict = dict(Counter(word_freq).most_common(10))
    
    for key,value in word_dict.items():
        word_counts.append(value)
        word_names.append(key)
        
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Genre Count"
                },
                'xaxis': {
                    'title': "Message Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=word_names,
                    y=word_counts
                )
            ],

            'layout': {
                'title': {
                    'text': '10 Most Occuring words',
                    'font': dict(
                        family="Times New Roman, serif",
                        size=18
                    )
                },
                'paper_bgcolor':'rgb(240, 248, 235)',
                'plot_bgcolor':'rgb(248, 243, 235)',
                'yaxis': {
                    'title': {
                        'text': "Word Count",
                        'font': dict(
                            family="Times New Roman, serif",
                            size=14,
                            color="#7f7f7f"
                        )
                    }
                },
                'xaxis': {
                    'title': {
                        'text': "Words",
                        'font': dict(
                            family="Times New Roman, serif",
                            size=14,
                            color="#7f7f7f"
                        )
                    }
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Message Category",
                    'tickangle': 45
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=False)


if __name__ == '__main__':
    main()