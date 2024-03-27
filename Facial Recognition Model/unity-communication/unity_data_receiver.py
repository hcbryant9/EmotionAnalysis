from flask import Flask, request, jsonify, Response
import csv
import os
import pandas as pd
import joblib
import numpy as np
import re
import json
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
FACE_PATH = 'data/face/data.txt'
STT_PATH = 'data/stt/data.txt'
SENTIMENT_PATH = 'data/sentiment/data.txt'

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

    # Write the prediction to a file for visualization
    with open('data/face/data.txt', 'a') as file:
        file.write(str(prediction) + '\n')

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



@app.route('/metrics')
def metrics():
    '''
    Expose Prometheus metrics including data received via STT, sentiment, and face predictions for visualization
    '''
    # Example metrics based on data counts
    stt_data_metric = get_last_stt(STT_PATH)
    sentiment_data_metric = get_last_sentiment(SENTIMENT_PATH)
    face_data_metric = get_last_face(FACE_PATH)

    # Construct Prometheus exposition format response
    metrics_text = (
        f"# HELP stt_data_metric Description of STT data metric\n"
        f"# TYPE stt_data_metric gauge\n"
        f"stt_data_metric {stt_data_metric}\n"
        f"# HELP sentiment_data_metric Description of sentiment data metric\n"
        f"# TYPE sentiment_data_metric gauge\n"
        f"sentiment_data_metric {sentiment_data_metric}\n"
        f"# HELP face_data_metric Description of face data metric\n"
        f"# TYPE face_data_metric gauge\n"
        f"face_data_metric {face_data_metric}\n"
    )

    # Return the response with the metrics text and appropriate content type
    return Response(metrics_text, mimetype='text/plain')


#This script makes a prediction
def make_prediction(data):
    rf_classifier = joblib.load('models/pickle/random_forest_model.pkl')

    new_data = pd.DataFrame([data])  #convert received data to a DataFrame
    proba_rf = rf_classifier.predict_proba(new_data) #return the value with the highest probability (that is the probability of the emotion predicted)
    prob_value_rf = np.max(proba_rf)
    # get the emotion
    # todo -> whatever has the highest probability -> index of 3 means sad, save a prediction. ?
    predicted_class_index_rf = rf_classifier.predict(new_data)[0]
    #todo -> add datavis
    return {
        'random_forest': {
            'emotion': predicted_class_index_rf,
            'probability': prob_value_rf
        }
    } 

# Function to get the last non-empty line of a file
def get_last_stt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        non_empty_lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
        if non_empty_lines:
            return non_empty_lines[-1]
        else:
            return ""
# Function to get the most recent sentiment data
def get_last_sentiment(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expression to find all entries within double quotes
    entries = re.findall('"([^"]+)"', content)

    # Return the most recent entry
    if entries:
        return entries[-1]
    else:
        return None

def get_last_face(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            # Get the last line in the file
            last_line = lines[-1].strip()
            # Assuming the data is stored in JSON format, you can parse it
            prediction_data = json.loads(last_line)
            # Extract emotion and probability from the prediction data
            emotion = prediction_data['random_forest']['emotion']
            probability = prediction_data['random_forest']['probability']
            return emotion, probability
        else:
            return None, None

if __name__ == '__main__':
    # Start HTTP server to expose metrics on port 8000 to prometheus
    start_http_server(8000)
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000)


