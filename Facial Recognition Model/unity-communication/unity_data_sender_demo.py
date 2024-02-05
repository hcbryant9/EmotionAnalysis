import requests

# URL of the Flask server
SERVER_URL = 'http://localhost:5000/receive_data'

# Sample JSON data to send
sample_data = ['0.031226','0.01600352','6.026572E-07','1.680299E-05','0.01230686','0.008413654','1.401298E-45','1.401298E-45','9.377176E-07','0.005434837','7.640086E-12','3.351888E-21','0.07911677','0.02501802','0','0','0.06836806','0.1574175','0','0','0.5512146','0.5487477','0.008257591','0.001661794','0.01790047','1.083152E-05','0.003993776','3.158729E-17','0.0001891693','2.195447E-06','0.009904319','0.00979636','1.102334E-05','5.602955E-06','1.156553E-21','1.753774E-06','1.28102E-21','8.238605E-07','2.778148E-07','2.778148E-07','1.109966E-06','1.030741E-06','0.005446971','0.005454915','0.09100145','0.003281127','0.08486544','0.0008876458','3.695598E-06','3.188127E-06','0.02055652','1.897524E-21','3.147027E-21','0.002685157','0.005650207','5.00322E-06','0.001768717','2.898303E-08','6.470232E-06','0.01066079','0.01375919','0.009043655','0.0239103','Sad']

# Function to send data to the Flask server
def send_data_to_server(data):
    try:
        # Send POST request to the Flask server
        response = requests.post(SERVER_URL, json=data)
        
        # Check if the response is successful
        if response.status_code == 200:
            # Parse the JSON response
            json_response = response.json()
            # Print the server's response
            print("Server response:", json_response)
        else:
            print("Server returned status code:", response.status_code)
            print("Server response:", response.text)
    except Exception as e:
        print("Error occurred:", str(e))

if __name__ == "__main__":
    # Send the sample data to the server
    send_data_to_server(sample_data)
