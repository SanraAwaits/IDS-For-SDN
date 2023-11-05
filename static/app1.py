import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


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
engine = create_engine('sqlite:///../data/Disaster_ETL.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/model.pkl")



app = Flask(__name__)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')
@app.route('/login')
def login():
    return render_template('login.html')
def home():
	return render_template('home.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df) 

@app.route('/prediction1')
def prediction1():
    return render_template('home.html')  
@app.route('/chart')
def chart():
    return render_template('chart.html')
@app.route('/prediction')
def prediction():
 	return render_template("home.html")
@app.route('/crime')
def crime():
 	return render_template("crime.html")
@app.route('/crimes')
def crimes():
 	return render_template("crimes.html")
@app.route('/total')
def total():
 	return render_template("total.html")
@app.route('/theft')
def theft():
    return render_template('theft.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("some.csv", encoding="latin-1")
	# Features and Labels
	df['label'] = df['l']
	X = df['t']
	y = df['label']
	print(X)
	# Extract Feature With CountVectorizer
	cv = CountVectorizer(ngram_range=(1, 2))
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB

	clf = svm.SVC(kernel='linear')
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)