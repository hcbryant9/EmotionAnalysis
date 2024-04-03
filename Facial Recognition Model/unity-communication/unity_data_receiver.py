from flask import Flask, request, jsonify, Response, make_response
import csv
import os
import pandas as pd
import joblib
import numpy as np
import re
import json
from prometheus_client import start_http_server, Counter, generate_latest
import warnings

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

# Define a dictionary to store emotion probabilities
sentiment_probability = {"Happy": 0.0, "Sad": 0.0, "Mad": 0.0, "Scared": 0.0}
face_probability = {"Happy": 0.0, "Sad": 0.0, "Mad": 0.0, "Anxious": 0.0}

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
    warnings.filterwarnings("ignore")
    prediction = make_prediction(data)
    warnings.filterwarnings("ignore")
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
    # Assuming data is in the format "Happy: 9/10 Sad: 1/10 Mad: 1/10 Scared: 1/10"
    emotions = data.split()
    emotion_values = [float(emotion.split('/')[0]) / 10 for emotion in emotions[1::2]]
    emotion_labels = ["Happy", "Sad", "Mad", "Scared"]  # Assuming this order of emotions
    
    # Update emotion probabilities
    for label, value in zip(emotion_labels, emotion_values):
        sentiment_probability[label] = value

    # Convert to JSON format
    json_data = []
    for label, value in sentiment_probability.items():
        json_data.append({"emotion": label.lower(), "probability": value})

    # Writing data to a .txt file
    file_path = 'data/sentiment/data.txt'
    with open(file_path, 'a') as file:
        for item in json_data:
            file.write(str(item) + '\n')

    response_data = {"message": "Voice emotion received successfully"}

    return jsonify(response_data)



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
    # Retrieve data metrics
    stt_data_metric = get_last_stt(STT_PATH)
    happy_sentiment_metric = face_probability["Happy"]
    
    
    # Construct Prometheus exposition format response
    metrics_text = (
        f"# HELP happy_face_metric Probability of Happy Emotion from Face Prediction Model"
        f"# TYPE happy_face_metric gauge\n"
        f"happy_face_metric {face_probability['Happy']}\n"
        f"# HELP sad_face_metric Probability of Sad Emotion from Face Prediction Model"
        f"# TYPE sad_face_metric gauge\n"
        f"sad_face_metric {face_probability['Sad']}\n"
        f"# HELP mad_face_metric Probability of Mad Emotion from Face Prediction Model"
        f"# TYPE mad_face_metric gauge\n"
        f"mad_face_metric {face_probability['Mad']}\n"
        f"# HELP anxious_face_metric Probability of Anxious Emotion from Face Prediction Model"
        f"# TYPE anxious_face_metric gauge\n"
        f"anxious_face_metric {face_probability['Anxious']}\n"
        f"# HELP happy_sentiment_metric Probability of Happy Emotion from STT"
        f"# TYPE happy_sentiment_metric gauge\n"
        f"happy_sentiment_metric {sentiment_probability['Happy']}\n"
        f"# HELP sad_sentiment_metric Probability of Sad Emotion from STT"
        f"# TYPE sad_sentiment_metric gauge\n"
        f"sad_sentiment_metric {sentiment_probability['Sad']}\n"
        f"# HELP mad_sentiment_metric Probability of Mad Emotion from STT"
        f"# TYPE mad_sentiment_metric gauge\n"
        f"mad_sentiment_metric {sentiment_probability['Mad']}\n"
        f"# HELP anxious_sentiment_metric Probability of Anxious Emotion from STT"
        f"# TYPE anxious_sentiment_metric gauge\n"
        f"anxious_sentiment_metric {sentiment_probability['Scared']}\n"
        
    )

    # Create a response with the metrics text
    response = make_response(metrics_text)
    response.mimetype = "text/plain"
    return response





#This script makes a prediction
def make_prediction(data):
    warnings.filterwarnings("ignore")
    rf_classifier = joblib.load('models/pickle/random_forest_model.pkl')

    new_data = pd.DataFrame([data])  #convert received data to a DataFrame
    proba_rf = rf_classifier.predict_proba(new_data) #return the value with the highest probability (that is the probability of the emotion predicted)
    prob_value_rf = np.max(proba_rf)
    
    # get the emotion
    predicted_class_index_rf = rf_classifier.predict(new_data)[0]
    
    # Update face probabilities
    for emotion in face_probability:
        if emotion == predicted_class_index_rf:
            face_probability[emotion] = prob_value_rf
        else:
            face_probability[emotion] = 0.0

    # Return prediction
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


