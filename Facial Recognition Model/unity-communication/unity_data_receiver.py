from flask import Flask, request, jsonify, Response
import csv
import os
import pandas as pd
import joblib
import numpy as np
from prometheus_client import start_http_server, Counter, generate_latest


#from visualization import update_emotion_probability, plot_emotion_graphs
''' 
This script acts as the main server for facial data collection and prediction
Initiate server by running unity_data_receiver.py in the terminal
Send a request to the server in Unity with the OVRFacialExpressions.toArray()
This script will store the data to a new file 'data_log.csv' the string must contain 62 facial expressions and 1 emotion classifer at the end
Recommended -> Rename generated csv file to format 'data_log_INITIALS(ITERATION)_MONTH_DAY.csv' (refer to /data for example)
The program will then make a prediction of what it thinks the emotion is and return it to Unity
The way this is done is that it opens 'single_emotion.csv' and replaces the 2nd line of the csv file with the most recent data sent
not including the specified emotion. Then sends this to the .pkl files found in models and returns the results.
'''
app = Flask(__name__)
# Define a counter metric for counting predictions
prediction_counter = Counter('predictions_total', 'Total number of predictions made')

# Route to receive data from the Unity game
@app.route('/receive_data', methods=['POST'])
def receive_data():
    '''
    This route is for collecting data only for facial data
    '''
    data = request.get_json()

    # Write the received data to a CSV file
    with open('data_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    # Respond to the client with successful message
    return jsonify({"message": "Data received successfully"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    '''
    This route is for making predictions on an unknown emotion. Not for collecting data.
    '''
    data = request.get_json()

    # Make prediction on the received data
    prediction = make_prediction(data)

    # Increment the prediction counter
    prediction_counter.inc()

    # Respond to the client with predictions dt and rf
    return jsonify(prediction), 200

@app.route('/predict_while_collect', methods=['POST'])
def predict_while_collect():
    '''
    This route was created to be able to predict while collecting data from user
    While collecting data, Unity sends the emotion appended to the data
    This route drops that emotion and returns the prediction
    '''
    data = request.get_json()

    new_data = data[:-1]

    prediction = make_prediction(new_data)
    

    return jsonify(prediction), 200

@app.route('/receive_voice_emotion', methods=['POST'])
def receive_voice_emotion():
    '''
    This route is for receiving the results of chat gpt emotion prediction
    The result will be stored and sent on to prometheus
    '''
    data = request.data.decode('utf-8')
    
    # Process data here, assuming it was successful
    response_data = {"message": "Voice emotion received successfully"}

    # Writing data to a .txt file in CSV format
    file_path = 'data/sentiment/data.txt'
    with open(file_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([data])
        
    return jsonify(response_data), 200

@app.route('/receive_stt', methods=['POST'])
def receive_stt():
    '''
    This route is for receiving the speech to text from WitAI in Unity
    The result will be stored and sent on to prometheus
    '''
    data = request.data.decode('utf-8')
    
    # Process data here, assuming it was successful
    response_data = {"message": "Speech to text received successfully"}

    # Writing data to a .txt file in CSV format
    file_path = 'data/stt/data.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([data])

    return jsonify(response_data), 200

    
# Route to expose Prometheus metrics
@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

#This script makes a prediction
def make_prediction(data):
    dt_classifier = joblib.load('models/pickle/decision_tree_model.pkl')
    rf_classifier = joblib.load('models/pickle/random_forest_model.pkl')

    new_data = pd.DataFrame([data])  #convert received data to a DataFrame

    #predict probabilities for each class
    proba_dt = dt_classifier.predict_proba(new_data)
    proba_rf = rf_classifier.predict_proba(new_data)

    #return the value with the highest probability (that is the probability of the emotion predicted)
    prob_value_dt = np.max(proba_dt)
    prob_value_rf = np.max(proba_rf)

    # get the emotion
    # todo -> whatever has the highest probability -> index of 3 means sad, save a prediction. ?
    predicted_class_index_dt = dt_classifier.predict(new_data)[0]
    predicted_class_index_rf = rf_classifier.predict(new_data)[0]

    #todo -> add datavis
    #update_emotion_probability(predicted_class_index_rf, prob_value_rf)
    #plot_emotion_graphs()
    return {
        'decision_tree': {
            'emotion': predicted_class_index_dt,
            'probability': prob_value_dt

        },
        'random_forest': {
            'emotion': predicted_class_index_rf,
            'probability': prob_value_rf
        }
    } 

    



if __name__ == '__main__':
    # Start HTTP server to expose metrics on port 8000 to prometheus
    start_http_server(8000)
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000)
