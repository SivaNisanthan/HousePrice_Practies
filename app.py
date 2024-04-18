import pickle
from flask import Flask, request, jsonify, render_template,url_for
import numpy as np
import pandas as pd

#starting point of my app where it is going to run
app = Flask(__name__)  # Initialize the flask App
regmodel = pickle.load(open('regmodel.pkl', 'rb'))  # Load the model
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler
@app.route('/')  # Homepage
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])  # To use the predict button in our web-app
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])  # To use the predict button in our web-app
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template('home.html', prediction_text='House Price should be $ {}'.format(output))



if __name__ == '__main__':
    app.run(debug=True)  # Debug = True will reload the server each time we make changes in the code