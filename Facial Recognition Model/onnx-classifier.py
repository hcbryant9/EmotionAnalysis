import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# Load the data
data = pd.read_csv('data/unity_collection.csv')

# Separate features and target
X = data.drop('Emotion', axis=1).values  # Convert to NumPy array immediately
y = data['Emotion'].values

# Ensure we have 63 input features
assert X.shape[1] == 63, f"Expected 63 features, but got {X.shape[1]}"

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Convert Random Forest model to ONNX
initial_type = [('float_input', FloatTensorType([None, 63]))]
onnx_model = convert_sklearn(rf_model, initial_types=initial_type)

# Save the ONNX model
onnx_model_path = 'facial_expression_model.onnx'
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"ONNX model saved to {onnx_model_path}")

# Evaluate the model
accuracy = rf_model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Print the mapping between encoded labels and original emotions
print("\nEmotion label mapping:")
for emotion, encoded_label in zip(le.classes_, le.transform(le.classes_)):
    print(f"{emotion}: {encoded_label}")

# Verify ONNX model functionality
print("\nVerifying ONNX model functionality:")
ort_session = ort.InferenceSession(onnx_model_path)

# Test with a few samples
for i in range(5):
    input_sample = X_test[i].reshape(1, -1).astype(np.float32)
    true_label = y_test[i]
   
    ort_inputs = {ort_session.get_inputs()[0].name: input_sample}
    ort_outs = ort_session.run(None, ort_inputs)
    predicted_label = ort_outs[0][0]
    predicted_emotion = le.inverse_transform([predicted_label])[0]
    true_emotion = le.inverse_transform([true_label])[0]
   
    print(f"\nSample {i+1}:")
    print(f"Input shape: {input_sample.shape}")
    print(f"Predicted emotion: {predicted_emotion}")
    print(f"True emotion: {true_emotion}")
    print(f"Prediction {'correct' if predicted_emotion == true_emotion else 'incorrect'}")

print("\nONNX model verification complete.")

# Demonstrate usage with 63 random facial points
print("\nDemonstrating usage with random facial points:")
random_facial_points = np.random.rand(1, 63).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: random_facial_points}
ort_outs = ort_session.run(None, ort_inputs)
predicted_label = ort_outs[0][0]
predicted_emotion = le.inverse_transform([predicted_label])[0]

print(f"Random input shape: {random_facial_points.shape}")
print(f"Predicted emotion: {predicted_emotion}")

print("\nScript execution complete.")