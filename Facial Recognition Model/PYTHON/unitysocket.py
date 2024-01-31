from flask import Flask, request

app = Flask(__name__)

# Route to receive data from the Unity game
@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()

    # Write the received data to a text file
    with open('data_log.txt', 'a') as file:
        file.write(str(data) + '\n')

    # Respond to the client
    return 'Data received successfully!', 200

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000)
