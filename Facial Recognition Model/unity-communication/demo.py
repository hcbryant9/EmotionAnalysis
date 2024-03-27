import csv
import os
import re
import json

FACE_PATH = 'data/face/data.txt'
STT_PATH = 'data/stt/data.txt'
SENTIMENT_PATH = 'data/sentiment/data.txt'

# Function to get the most recent sentiment data
def get_most_recent_entry(file_path):
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
    
def extract_recent_prediction(file_path):
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

# Function to get the last non-empty line of a file
def get_last_stt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        non_empty_lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
        if non_empty_lines:
            return non_empty_lines[-1]
        else:
            return ""


# Get the most recent sentiment data
recent_sentiment = get_most_recent_entry(SENTIMENT_PATH)
recent_face = extract_recent_prediction(FACE_PATH)
recent_stt = get_last_stt(STT_PATH)
print(recent_sentiment)
print(recent_face)
print(recent_stt)