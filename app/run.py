import json
import plotly
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    ''' Usage: normalize case and remove punctuation
        Args: text string
        Return: text tokens
    ''' 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('msg_cat', engine)
# load saved model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    Usage: format and display 4 plots: 
           1. count distribution of message genres, 
           2. percentage distribution of message genres
           3. count distribution of disaster categories
           4. percentage distribution of disaster categories
    Return: a block of dynamic html codes that can be rendered to the plots
    '''
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    cat1_counts = df.iloc[:, 4:].sum().sort_values(ascending = False)
    cat0_counts = df.shape[0] - cat1_counts
    cat1_pcts = cat1_counts/df.shape[0]*100
    cat0_pcts = 100 - cat1_pcts
    cat_names = list(cat0_counts.index)

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
                'autosize': True,
                'title': 'Count Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'autosize': True,
                'title': 'Percentage Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat0_counts,
                    name='value: 0'
                ),
                Bar(
                    x=cat_names,
                    y=cat1_counts,
                    name='value: 1'
                )
            ],

            'layout': {
                'autosize': True,
                'title': 'Count Distribution of Disaster Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat0_pcts,
                    name='value: 0'
                ),
                Bar(
                    x=cat_names,
                    y=cat1_pcts,
                    name='value: 1'
                )
            ],

            'layout': {
                'autosize': True,
                'barmode': 'stack',
                'title': 'Percentage Distribution of Disaster Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
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
    '''
    Usage: User queries an input of text from the front end to backend. 
           The text is then processed and fed into trained ML model to predict categories.
    Return: a block of dynamic html codes that can be rendered to show a list of predicted classified categories
    '''
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
    '''
    This is the main entry to start the web app.
    '''
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()