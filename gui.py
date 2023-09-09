from flask import Flask, url_for, render_template, redirect, request, session
import sys
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model, model_from_json
import keras
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.framework import ops
from joblib import load
from keras.preprocessing.sequence import pad_sequences

graph = ops.get_default_graph()

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

app = Flask(__name__)
app.secret_key = "dim-vas"

LR_pipeline = load("MyModels/LR_model.joblib")
DT_pipeline = load("MyModels/DT_model.joblib")
SVM_pipeline = load("MyModels/SVM_model.joblib")
tokenizer_pkl = open('MyModels/tokenizer.pkl', 'rb')
tokenizer = pickle.load(tokenizer_pkl)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=["POST", "GET"])
def form():
    if request.method == "GET":
        return render_template('form.html') 
    elif request.method == "POST":
        Tweet = request.form["Tweet"]
        session["Tweet"] = Tweet
        return redirect(url_for('model')) 

@app.route('/model', methods=["POST", "GET"])
def model():
    if request.method == "GET":
        return render_template('model.html')
    elif request.method == "POST":
        mdl = request.form["model"]
    return redirect(url_for('result', mdl = mdl))

@app.route('/<mdl>')
def result(mdl):
    if "Tweet" in session:
        Tweet = session["Tweet"]
        case = [Tweet]                       
      
    if(mdl == "Logistic Regression"):
        clf = LR_pipeline 
    elif(mdl == "Decision Tree Classifier"):
         clf = DT_pipeline
    elif(mdl == "SVM"):
        clf = SVM_pipeline
    elif(mdl == "CNN"):
        case = tokenizer.texts_to_sequences(Tweet)
        case = pad_sequences(case, padding='post', maxlen=1000)
        #global graph
        with graph.as_default():
            CNN_pkl = open('MyModels/CNN.pkl', 'rb')
            CNN_model = pickle.load(CNN_pkl)
            predictions = CNN_model.predict(case)
            predictions = (predictions > 0.5)
            if(predictions.any()):
                prediction = 1
            else:
                prediction = 0 
    
    if(mdl != "CNN") :
        prediction = clf.predict(case) 
            
    if(prediction==1):
        output = "Positive"
    else:
        output = "Negative"
    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run()