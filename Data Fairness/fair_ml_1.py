
# pip install joblib aif360
import joblib
import pandas as pd
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# Load the pre-trained model
model = joblib.load('trained_model.pkl')

# Example features (e.g., CPU usage, packet size, etc.)
new_data = [[35, 1500, 0.05]]  # Example feature data

# Predict with the loaded model
prediction = model.predict(new_data)
print(f"Prediction: {prediction}")

# Simulate a dataset with a sensitive attribute (e.g., gender, race)
data = {
    'cpu_usage': [30, 40, 50, 60, 70],
    'packet_size': [1000, 1500, 2000, 2500, 3000],
    'anomaly_score': [0.1, 0.2, 0.3, 0.4, 0.5],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],  # Sensitive attribute
    'label': [0, 1, 0, 1, 0]  # Model prediction (0: normal, 1: anomaly)
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert to a StandardDataset object (required by AIF360)
dataset = StandardDataset(df, label_name='label', protected_attribute_names=['gender'], 
                           privileged_classes=[['Male']])

# Display dataset
print(dataset.features[:5])

# Convert predictions into a dataset
predictions = model.predict(dataset.features)

# Create a dataset with predictions
dataset_with_predictions = dataset.copy()
dataset_with_predictions.labels = predictions

# Calculate fairness metrics
metric = BinaryLabelDatasetMetric(dataset_with_predictions, 
                                  privileged_groups=[{'gender': 'Male'}],
                                  unprivileged_groups=[{'gender': 'Female'}])

# Fairness metrics
print(f"Disparate Impact: {metric.disparate_impact()}")
print(f"Mean Difference: {metric.mean_difference()}")

# Apply reweighing to mitigate bias
reweigher = Reweighing(privileged_groups=[{'gender': 'Male'}], unprivileged_groups=[{'gender': 'Female'}])
dataset_transformed = reweigher.fit_transform(dataset)

# Display transformed dataset
print(dataset_transformed.feature_names[:5])

# Retrain the model using the transformed dataset
X_transformed = dataset_transformed.features
y_transformed = dataset_transformed.labels

model.fit(X_transformed, y_transformed)

# Predict on the transformed data
new_prediction = model.predict(new_data)
print(f"New Prediction: {new_prediction}")