from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import model and scaler pkl files
model = pickle.load(open('./model/ridge.pkl', 'rb'))
scaler = pickle.load(open('./model/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predictdata():
    if request.method == 'POST':
        try:
            Temparature = float(request.form.get('Temparature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            new_scaled_data = scaler.transform([[Temparature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = model.predict(new_scaled_data)

            return render_template('home.html', results=result[0])

        except Exception as e:
            print(f"Error: {e}")
            return render_template('home.html', results='Error occurred during prediction.')

    else:
        return render_template('home.html')

    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)