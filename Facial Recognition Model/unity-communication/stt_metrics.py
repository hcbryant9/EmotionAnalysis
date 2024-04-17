from prometheus_client import start_http_server, Counter, generate_latest, Gauge
from flask import Flask, request, jsonify, Response, make_response
import os
app = Flask(__name__)

STT_PATH = 'data/stt/data.txt'


@app.route('/metrics')
def metrics():
    with open(STT_PATH, 'r') as file:
        lines = file.readlines()
        non_empty_lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
        if non_empty_lines:
            return non_empty_lines[-1]
        else:
            return ""
        

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5005)