#This file makes predictions based off the train .onnx file & is a slight variation of predicition-onnx.py
import pandas as pd
import onnxruntime as ort

# Load ONNX models
dt_model = ort.InferenceSession('models/decision_tree_model.onnx')
rf_model = ort.InferenceSession('models/random_forest_model.onnx')

# Assuming it's in a format similar to the training data
new_data = pd.read_csv('data/sample_single_emotion.csv')

# Convert new_data to a numpy array
new_data_np = new_data.values.astype('float32')

# Make Predictions
new_predictions_dt = dt_model.run(None, {'float_input': new_data_np})
new_predictions_rf = rf_model.run(None, {'float_input': new_data_np})

# Print predictions
print("Decision Tree Predictions:", new_predictions_dt)
print("Random Forest Predictions:", new_predictions_rf)
