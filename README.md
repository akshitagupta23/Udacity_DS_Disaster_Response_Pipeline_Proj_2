# Disaster Response Pipeline Project
This Machine Learning project is also part of a Nanodegree in Data Science at Udacity.

#### Business Problem:
- To classify disaster labeled data from `Figure Eight` which were sent during disasters.
- Classification of these messages, allows these messages to be sent to disaster relief agency using an web app. 

#### Data and Methodology
Data set consists of 36 pre-defined categories and messages in `categories.csv` and `messages.csv` respectively. 
Methodology of building the `web app` involved following steps:
- 1.ETL pipeline that cleans the data and  store in Database
- 2.Machine Learning pipeline that builds features using NLP techniques and Multiclassifier Algorithm for training. Optimization of Algorithm was performed using `GridSearchCV`
- 3.Web application build using html templates and `flask` api was used to get prediction of new messages and to visualize training data insights.


#### Project Structure
    
            ├── app
            │   ├── run.py
            │   └── templates
            │       ├── go.html
            │       └── master.html
            ├── data
            │   ├── disaster_categories.csv
            │   ├── disaster_messages.csv
            │   ├── DisasterResponse.db
            │   └── process_data.py
            ├── logs
            │   ├── disaster_response_etl.log
            │   └── disaster_response_ml_gridcv.log
            ├── models
            │   ├── classifier.pkl
            │   └── train_classifier.py
            └── README.md

#### Instructions:

- 1.  Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- 2. Run the following command in the app's directory to run your web app.
    `python run.py`

- 3. Go to http://0.0.0.0:3001/

#### Licensing, Authors, and Acknowledgements

Credit goes to Figure Eight and Udacity for providing the dataset.
The project is under MIT License.
