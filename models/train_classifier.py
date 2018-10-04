import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, make_scorer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

def load_data(database_filepath):
    '''
    Usage: load data from database
    Args: file path to the database
    Return: X - dataframe of features
            y - dataframe of labels
    '''
    db_name = database_filepath
    tbl_name = 'msg_cat'
    print('Opening connection.')
    conn = sqlite3.connect(db_name)
    # get a cursor
    cur = conn.cursor()
    cmd = "SELECT * FROM " + tbl_name
    print('Reading data from table "{}" in the database "{}".'.format(tbl_name, db_name))
    df = pd.read_sql(cmd, con = conn)
    df.set_index('id', inplace = True)
    conn.commit()
    conn.close()
    print('Connection is closed.')
    feature_columns = ['message', 'original', 'genre']
    X = df['message'].values
    y = df.drop(labels = feature_columns, axis = 1).values
    category_names = df.columns.values
    return X, y, category_names


def tokenize(text):
    ''' Usage: normalize case and remove punctuation
        Args: text string
        Return: text tokens
    ''' 
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Reduce words to their root form
    tokens = [WordNetLemmatizer().lemmatize(token).strip() for token in tokens]

    return tokens


def build_model():
    """
    Usage: builds classification model 
    Args:
    Return: the optimized model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(LinearSVC()))
    ])
    
    parameters = {
        'features__text_pipeline__vect__max_df': [0.2],
        'features__text_pipeline__vect__max_features': [None],
        'clf__estimator__C': [0.5],
        'clf__estimator__intercept_scaling': [1.5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs = 2, verbose = 2)
    return cv

def evaluate_model(model, X_test, y_test, target_names, metrics = None):
    '''
        Usage: show metrics with options of accuracy, f1, precision, recall scores.
        Args: y_test, y_pred - actual label values, predicted labels
               target_names - a list of labels
               metrics - available options ['accuracy', 'f1', 'precision', 'recall']
                         other option: show f1, precision, recall all together.
    '''
    
    y_pred = model.predict(X_test)
    
    for idx_col in range(y_test.shape[1]):
        if metrics == 'accuracy':
            # accuracy score
            print("The accuracy score for column {}: {}" \
                  .format(target_names[idx_col], accuracy_score(y_test[:, idx_col], y_pred[:, idx_col])))
        elif metrics == 'f1':
            # f1 score
            print("The f1 score for column {}: {}" \
                  .format(target_names[idx_col], f1_score(y_test[:, idx_col], y_pred[:, idx_col])))
        elif metrics == 'precision':
            # precision
            print("The precision score for column {}: {}" \
                  .format(target_names[idx_col], precision_score(y_test[:, idx_col], y_pred[:, idx_col])))
        elif metrics == 'recall':
            # precision
            print("The recall score for column {}: {}" \
                  .format(target_names[idx_col], recall_score(y_test[:, idx_col], y_pred[:, idx_col])))
        else:
            print(classification_report(y_test, y_pred, target_names = target_names))
            break


def save_model(model, model_filepath):
    """
    Usage: Save the model to a Python pickle
    Args:
        model: Trained model
        model_filepath: Path where to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # drop child_alone category in the label set
        #Y.drop(labels = 'child_alone', axis = 1, inplace = True)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()