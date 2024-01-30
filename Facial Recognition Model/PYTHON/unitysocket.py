from flask import Flask, request

app = Flask(__name__)

# Route to receive data from the Unity game
@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()

    # Do something with the received data
    print("Received data from Unity:", data)

    # You can process the data here as needed
    # For example, you can store it in a database, write to a file, etc.

    return 'Data received successfully!', 200

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000)
