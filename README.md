# disaster_response_pipelines_project
Machine Learning ETL pipeline for Figure8 disaster response classification modelling and web app

## Motivation

In this project we analyze disaster data from Figure Eight and build a model that classifies disaster messages. The data set contains real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agencyThe project include these three components (ETL Pipeline, Machine learning Pipeline, and web application) that uses a trained model and classify any messages. The machine learning pipeline created is assigned to categorize the events to send the messages to an appropriate disaster relief agency. The web app will also display visualizations of the data.

## Required libraries

- nltk 
- numpy
- pandas 
- scikit-learn 
- sqlalchemy 

Run the following commands in the project's root directory to set up your database and model.
To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse1.db models/model_classifier.pkl
Run the following command in the app's directory to run your web app. run.py

Go to http://0.0.0.0:3001/ to view the web application, use 'env|grep WORK' command to find the exact space ID to add to your , as anexample that would be: "https://viewa7a4999b-3001.udacity-student-workspaces.com/" 

Below are a few screenshots of the web app.

<img src="https://github.com/maggiehsuan/disaster_response_pipelines_project/blob/master/workspace/png/disaster-response-project1.png" width="60%" alt="disaster response project web app">
<img src="https://github.com/maggiehsuan/disaster_response_pipelines_project/blob/master/workspace/png/disaster-response-project2.png" width="60%" alt="disaster response project web app">

## File Descriptions

1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
- Add data visualizations using Plotly in the web app. 
- Run.py	Flask Web App (display visualization from the datasets, the app accept messages from users and returns classification results for 36 categories of disaster events)

## Licensing, Authors, Acknowledgements

This app is part of the Udacity Data Scientist Nanodegree. The data was provided by Figure Eight.