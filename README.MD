# Udacity Data Science Nanodegree
## Data Engineering Project: Disaster-Response-Pipeline

### To view the number of commits, as required by Udacity grading rubric, you can follow my Github link [Here](https://github.com/Zhitaow/Disaster-Response-Pipeline.git)

### Project Overview
In this course, I've learned and built on my data engineering skills to expand my opportunities and potential as a data scientist. In this project, I applied these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

This repository contains data set of real messages that were sent during disaster events, courtesty by [Figure Eight](https://www.figure-eight.com/data-for-everyone/). My motivation is to create a machine learning pipeline to categorize these events so that user can send the messages to an appropriate disaster relief agency.

My project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

The animation below is a demo of my web app.
![Alt text](/img/ScreenDemo.gif)

### Project Components
There are three major parts in this project: 

1. ETL Pipeline
The Python script, **process_data.py**, runs a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets, disaster_messages.csv and disaster_categories.csv
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
The Python script, **train_classifier.py**, runs a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
This web application designed with bootstrap to:

- process and visualize the cleaned data using Plotly
- classifer message from user input, and display the predicted disaster categories


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# File Descriptions <a name="files"></a>
The development of this project contains four folders corresponding to different stage in development:

### /dev contains:

**ETL Pipeline Preparation.ipynb**
- A Jupyter Notebook with all the steps to developing ETL pipeline.

**ML Pipeline Preparation.ipynb**
- A Jupyter Notebook with all the steps to developing machine learning pipeline.

**helper.py**
- A python script that provide a few helper functions for exploratory analysis in the notebook files.

### /data contains:

**process_data.py**
an ETL pipeline that cleans the data and prepare it for training the ML model:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

**disaster_messages.csv**
- A csv file containing the messages during the disaster
**disaster_categories.csv**
- A csv file containing 30 disaster types of messages

### /models contains:
**train_classifier.py**
a machine learning pipeline that trains the classifier using pipeline and gridsearch, and finally save the optimal model to a pickle file:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### /app
**run.py**
- A flask app that load and process the data, and render the contents into dynamic webpage.
**templates/master.html** 
- The index webpage where the major contents, four graphs of data distribution, will be rendered.
**templates/go.html**
- A segment of extension codes, showing list of classified message type, to be inserted to master.html.




