# DisasterResponsePipeline

This project will give a complete pipeline of analyzing messages sent during a disater

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Structure](#structure)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

For this project, I was interestested in building the entire workflow behind a meaningful project/app: build ETL to extract, transform and load data; build machine learning pipeline to train the model based on the data; and then feed the results into the app using flask, CSS and Javascript. Though I'm not an expert in any part of the three components, being able to combine all three pieces together under guidance made me proud of myself.

## Structure <a name="structure"></a>

The project contains three parts: data (ETL pipeline), models (ML pipeline) and app:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

## Results<a name="results"></a>

Results should be accessible through the web app.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the data. Also thanks for Udacity to provide the framework for me to finish the project.
