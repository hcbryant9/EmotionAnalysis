from flask import Flask, request, jsonify
import csv
import pandas as pd
import joblib
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

# Route to receive data from the Unity game
@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()

    # Write the received data to a CSV file
    with open('data_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    # Respond to the client with successful message
    return jsonify({"message": "Data received successfully"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Make prediction on the received data
    prediction = make_prediction(data)

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

#This script makes a prediction of the data stored in the second line in the csv file 'single_emotion.csv'
def make_prediction(data):
    
 
    dt_classifier = joblib.load('models/pickle/decision_tree_model.pkl')
    rf_classifier = joblib.load('models/pickle/random_forest_model.pkl')

    new_data = pd.DataFrame([data]) #convert received data to dataframes

    
    new_predictions_dt = dt_classifier.predict(new_data)
    new_predictions_rf = rf_classifier.predict(new_data)


    print("Decision Tree Predictions:", new_predictions_dt)
    print("Random Forest Predictions:", new_predictions_rf)

    return {
    'decision_tree_predictions': new_predictions_dt.tolist(),
    'random_forest_predictions': new_predictions_rf.tolist()
}

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000)
