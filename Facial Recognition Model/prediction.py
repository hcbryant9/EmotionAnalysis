import pandas as pd
import joblib

# Load trained models
dt_classifier = joblib.load('decision_tree_model.pkl')
rf_classifier = joblib.load('random_forest_model.pkl')

# Example new_data (replace this with your actual new data for prediction)
# Assuming it's in a format similar to the training data
new_data = pd.read_csv('sample_single_emotion.csv')

# Make Predictions
new_predictions_dt = dt_classifier.predict(new_data)
new_predictions_rf = rf_classifier.predict(new_data)

# Print predictions
print("Decision Tree Predictions:", new_predictions_dt)
print("Random Forest Predictions:", new_predictions_rf)
