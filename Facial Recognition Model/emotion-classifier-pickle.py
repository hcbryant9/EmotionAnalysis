# This file creates an ML model for .pkl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the CSV Data
data = pd.read_csv('data/unity_collection.csv')

# Step 2: Preprocess the Data
X = data.drop('Emotion', axis=1)  # Features (all landmarks)
y = data['Emotion']  # Labels (emotions)

# Step 3: Split the Data into Training and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Models

# Decision Tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Save the trained Decision Tree model
joblib.dump(dt_classifier, 'decision_tree_model.pkl')

# Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Save the trained Random Forest model
joblib.dump(rf_classifier, 'random_forest_model.pkl')

# Step 5: Make Predictions and Evaluate the Models

# Predictions
dt_predictions = dt_classifier.predict(X_test)
rf_predictions = rf_classifier.predict(X_test)

# Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))

print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_predictions))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
