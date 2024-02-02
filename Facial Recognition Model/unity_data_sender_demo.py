import requests
import json
import time

# URL of the Flask server
SERVER_URL = 'http://localhost:5000/receive_data'

# Sample JSON data to send
sample_data = ['Happy','Sad']
# Main function to send data to the Flask server
def send_data_to_server(data):
    try:
        # Send POST request to the Flask server
        response = requests.post(SERVER_URL, json=data)
        # Print the server's response
        print("Server response:", response.text)
    except Exception as e:
        print("Error occurred:", str(e))

if __name__ == "__main__":
    # Send the sample data to the server
    send_data_to_server(sample_data)
