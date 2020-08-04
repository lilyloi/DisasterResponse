# Disaster Response Pipeline Project

![Intro Pic](img/head.png)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installing)
	3. [Executing Program](#executing)
	4. [Additional Material](#material)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Screenshots](#screenshots)

<a name="descripton"></a>
## Description

This Project is a part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.
The initial dataset contains pre-labelled tweet and messages from real-life disaster. 
The aim of the project is to build a Natural Language Processing (NLP) tool that categorize disaster messages.

The Project is divided in the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper database structure.
2. Machine Learning Pipeline to train a model able to classify text messages into appropriate categories.
3. Web App to show model results in real time. 

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+ (I used Python 3.8)
* Machine Learning libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process libraries: NLTK
* SQLite Database libraries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

<a name="installing"></a>
### Installing
Clone this GIT repository:
```
git clone https://github.com/lng15/DisasterResponse
```
<a name="executing"></a>

### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train-classifier.py data/DisasterResponse.db model/DS_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="material"></a>

### Additional Material

In the **data** and **model** folder you can find Jupyter notebooks that will help you understand how the model works step by step:
1. **ETL Preparation Notebook**: an implemented ETL pipeline which extracts, transforms, and load raw dataset into a cleaned dataset. 
2. **ML Pipeline Preparation Notebook**: analyzing machine learning models through NLP process to find the final model.
3. **ML Pipeline Final Notebook**: the final machine learning model used for web app.

You can use **ML Pipeline Preparation Notebook** to re-train the model or tune it through Grid Search section.
In this case, it is warmly recommended to use a Linux machine to run Grid Search, especially if you are going to try a large combination of parameters.
Using a standard desktop/laptop (4 CPUs, RAM 8Gb or above) it may take several hours to complete. 

<a name="authors"></a>
## Authors

* [LN](https://github.com/lng15)

<a name="license"></a>

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>

## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing such a complete Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing messages dataset to train my model

<a name="screenshots"></a>
## Screenshots

1. This is an example of a message you can type to test Machine Learning model performance

![Sample Input](img/example_input.png)

2. After clicking **Classify Message**, you can see the categories which the message belongs to highlighted in green

![Sample Output](img/example.png)

3. The web app' s main page shows some graphs as an overview of training data.

![Main Page](img/overview.png)

4. The main page also displays the distribution of disaster messages from training data.

![Main Page](img/distribution.png)