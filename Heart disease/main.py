#import datetime
from flask import Flask, render_template, request, redirect, url_for
from google.cloud import datastore
import google.oauth2.id_token
from google.auth.transport import requests
import joblib
import sklearn
import numpy as np
import pandas as pd

model = joblib.load('knn_model.joblib')

app = Flask(__name__)

# get access to the datastore client so we can add and store data in the datastore
datastore_client = datastore.Client()

# get access to a request adapter for firebase as we will need this to authenticate users
firebase_request_adapter = requests.Request()

def retrieveUserInfo(claims):
 entity_key = datastore_client.key('UserInfo', claims['email'])
 entity = datastore_client.get(entity_key)
 return entity

def createUserInfo(claims):
 entity_key = datastore_client.key('UserInfo', claims['email'])
 entity = datastore.Entity(key = entity_key)
 entity.update({
 'email': claims['email'],
 #'name': claims['name']
 })
 datastore_client.put(entity)

@app.route('/')
def root():
    # query firebase for the request token and set other variables to none for now
    id_token = request.cookies.get("token")
    error_message = None
    claims = None
    #times = None

    # if we have an ID token then verify it against firebase if it doesn't check out then
    # log the error message that is returned
    if id_token:
        try:
            claims = google.oauth2.id_token.verify_firebase_token(id_token, firebase_request_adapter)

            user_info = retrieveUserInfo(claims)
            if user_info == None:
                createUserInfo(claims)
                user_info = retrieveUserInfo(claims)

        except ValueError as exc:
            error_message = str(exc)

    # render the template with the last times we have
    return render_template('index.html', user_data=claims, error_message=error_message)

# def createClient(name, surname, sex, age, bmi, race, smoke, alcohol, stroke, diabetes, asthma, kidneydisease, skincancer, diffwalking, physicalactivity, physicalhealth, mentalhealth, sleeptime):
#    entity_key = datastore_client.key('Client')
#    entity = datastore.Entity(key = entity_key)
#    entity.update({
#       'Name': name,
#       'Surname': surname,
#       'Sex': sex,
#       'Age': age,
#       'BMI': bmi,
#       'Race': race,
#       'Smoke': smoke,
#       'Alcohol': alcohol,
#       'Stroke': stroke,
#       'Diabetes': diabetes,
#       'Asthma': asthma,
#       'Kidneydisease': kidneydisease,
#       'Skincancer': skincancer,
#       'Diffwalking': diffwalking,
#       'PhysicalActivity': physicalactivity,
#       'PhysicalHealth': physicalhealth,
#       'MentalHealth': mentalhealth,
#       'SleepTime': sleeptime
#    })
#    return entity

# @app.route('/add_client', methods=['POST'])
# def addClient():
#         if request.method== 'POST':
#             entity = createClient((request.form['name']), (request.form['surname']), (request.form['sex']), (request.form['race']), (request.form['smoke']), (request.form['alcohol']), 
#                                   (request.form['stroke']), (request.form['diabetes']), (request.form['asthma']),(request.form['kidneydisease']), (request.form['skincancer']), 
#                                   (request.form['physicalactivity']), (request.form['physicalhealth']), (request.form['mentalhealth']), (request.form['sleeptime']), 
#                                   (request.form['diffwalking']), (request.form['bmi']), (request.form['age']))
            
#             datastore_client.put(entity)
#         else:
#             return render_template('index.html')    
#         return redirect('/')
#print("Prediction")
#@app.route('/', methods=['GET'])
#def index():
 #   return render_template('index.html')

@app.route('/client', methods = ['GET', 'POST'])
def features():
    if request.method == 'POST':
        #print("Prediction")
        #name = request.form['name']
        #surname = request.form['surname']
        sex = request.form['sex']
        age = (request.form['age'])
        bmi = float(request.form['bmi'])
        #race = request.form['race']
        smoke = request.form['smoke']
        alcohol = request.form['alcohol']
        stroke = request.form['stroke']
        diabetes = request.form['diabetes']
        asthma = request.form['asthma']
        kidneydisease = request.form['kidneydisease']
        skincancer = request.form['skincancer']
        diffwalking = request.form['diffwalking']
        physicalactivity = request.form['physicalactivity']
        physicalhealth = request.form['physicalhealth']
        mentalhealth = request.form['mentalhealth']
        #sleeptime = request.form['sleeptime']
        genhealth = request.form['genhealth']

        data = {
            'sex': [sex],
            'age': [age],
            'bmi': [bmi],
            #'race': [race],
            'smoke': [smoke],
            'alcohol': [alcohol],
            'stroke': [stroke],
            'diabetes': [diabetes],
            'asthma': [asthma],
            'kidneydisease': [kidneydisease],
            'skincancer': [skincancer],
            'diffwalking': [diffwalking],
            'physicalactivity': [physicalactivity],
            'physicalhealth': [physicalhealth],
            'mentalhealth': [mentalhealth],
            #'sleeptime': [sleeptime],
            'genhealth': [genhealth]
        }

        df = pd.DataFrame(data)

        df_encoded = pd.get_dummies(df, columns=['sex', 'age', 'bmi', 'smoke', 'alcohol', 'stroke', 'diabetes', 'asthma',
                                                  'kidneydisease', 'skincancer', 'diffwalking', 'physicalactivity', 'physicalhealth', 'mentalhealth', 'genhealth'])
        
        #encoded_feat = df_encoded.columns.tolist()
        
        #if len(encoded_feat) != len(expected_feat):
         #   return "Invalid entry"

        c_feat = df_encoded.to_numpy()

        prediction = model.predict(c_feat)

        print("Prediction", prediction)

        return redirect(url_for('display_results', prediction = prediction))
    return render_template('index.html')

@app.route('/results/<int:prediction>')
def display_results(prediction):
    prediction_txt = 'Positive' if prediction == 1 else 'Negative'
    return render_template('results.html', prediction=prediction_txt)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
