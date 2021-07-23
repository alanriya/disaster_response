# Disaster Response Pipeline Project

### Summary:
This project is a Disaster Response Pipeline Project that helps stakeholder to determine the natural disaster happening given the message sent.

To accomplish this, a machine learning pipeline is build to tokenize words and transforming them to variables can be used in machine learning classifiers.

For this project, random forest classifier is built.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files in the repositiory

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # script to merge the data, clean and save to DB
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py # script for the building the machine pipeline to classify messages.
|- classifier.pkl # saved model
README.md
