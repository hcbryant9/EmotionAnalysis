import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Step 1: Load the CSV Data
data = pd.read_csv('data/unity_collection.csv')

# Step 2: Preprocess the Data
X = data.drop('Emotion', axis=1)  # Features (all landmarks)
y = data['Emotion']  # Labels (emotions)

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Logistic Regression Model
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

# Convert and Save the trained Logistic Regression model as ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(lr_classifier, initial_types=initial_type)
with open("logistic_regression_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
