# This file creates an ML model for .onnx and is a slight variation of the emotion-classifier-pickle.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Step 1: Load the CSV Data
data = pd.read_csv('data/unity_collection.csv')

# Step 2: Preprocess the Data
X = data.drop('Emotion', axis=1)  # Features (all landmarks)
y = data['Emotion']  # Labels (emotions)

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Models

# Decision Tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Convert and Save the trained Decision Tree model as ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(dt_classifier, initial_types=initial_type)
with open("decision_tree_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Convert and Save the trained Random Forest model as ONNX
onnx_model = convert_sklearn(rf_classifier, initial_types=initial_type)
with open("random_forest_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
