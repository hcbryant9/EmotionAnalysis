from flask import Flask, request
import csv

app = Flask(__name__)

# Route to receive data from the Unity game
@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()

    # Write the received data to a CSV file
    with open('data_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    # Respond to the client
    return 'Data received and logged successfully!', 200

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000)
