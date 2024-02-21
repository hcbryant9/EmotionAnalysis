#This file makes predictions based off the train .pkl file
import pandas as pd
import joblib

# Load trained models
dt_classifier = joblib.load('models/pickle/decision_tree_model.pkl')
rf_classifier = joblib.load('models/pickle/random_forest_model.pkl')

# Example new_data (replace this with your actual new data for prediction)
# Assuming it's in a format similar to the training data
new_data = pd.read_csv('data/sample_single_emotion_josiah.csv')

# Make Predictions
new_predictions_dt = dt_classifier.predict(new_data)
new_predictions_rf = rf_classifier.predict(new_data)

# For models that support probability estimates (like Random Forests)
# Get probability estimates for each class
if hasattr(dt_classifier, 'predict_proba'):
    dt_proba = dt_classifier.predict_proba(new_data)
    print("Decision Tree Probability Estimates:", dt_proba)

if hasattr(rf_classifier, 'predict_proba'):
    rf_proba = rf_classifier.predict_proba(new_data)
    print("Random Forest Probability Estimates:", rf_proba)

# Print predictions
print("Decision Tree Predictions:", new_predictions_dt)
print("Random Forest Predictions:", new_predictions_rf)
