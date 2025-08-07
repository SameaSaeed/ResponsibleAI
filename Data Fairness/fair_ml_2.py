import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.inprocessing import ExponentiatedGradient

# Simulated dataset (security-related data: 1 for anomaly, 0 for normal traffic)
data = {
    'feature1': [0.1, 0.2, 0.3, 0.5, 0.9, 1.2, 1.3],
    'feature2': [10, 20, 30, 50, 80, 100, 120],
    'label': [0, 0, 0, 1, 1, 1, 0]  # 0 = normal, 1 = anomaly
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['feature1', 'feature2']]
y = df['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Here, we'll simulate the data with a protected attribute, e.g., 'gender' as a binary feature
data = {
    'feature1': [0.1, 0.2, 0.3, 0.5, 0.9, 1.2, 1.3],
    'feature2': [10, 20, 30, 50, 80, 100, 120],
    'gender': [1, 0, 1, 1, 0, 0, 1],  # Protected attribute: gender (1 = male, 0 = female)
    'label': [0, 0, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# Create a dataset object for AIF360
dataset = StandardDataset(df, label_name='label', protected_attribute_names=['gender'],
                          privileged_classes=[[1]], categorical_features=[], features_to_drop=[])

# Calculate fairness metrics
metric = ClassificationMetric(dataset, dataset, unprivileged_groups=[{'gender': 0}], privileged_groups=[{'gender': 1}])

print("Disparate Impact:", metric.disparate_impact())

print("Statistical Parity Difference:", metric.statistical_parity_difference())
print("Equal Opportunity Difference:", metric.equal_opportunity_difference())

# Apply Disparate Impact Remover to mitigate bias in training data
dir = DisparateImpactRemover()
transformed_data = dir.fit_transform(dataset)

# Retrain model with transformed data
transformed_X = transformed_data.features
transformed_y = transformed_data.labels
model.fit(transformed_X, transformed_y)

# Apply Exponentiated Gradient to modify the model training process
exp_grad = ExponentiatedGradient()
exp_grad.fit(dataset)

# Recalculate fairness metrics after adjustments
adjusted_metric = ClassificationMetric(transformed_data, transformed_data,
                                       unprivileged_groups=[{'gender': 0}],
                                       privileged_groups=[{'gender': 1}])

print("Disparate Impact after adjustments:", adjusted_metric.disparate_impact())
print("Statistical Parity after adjustments:", adjusted_metric.statistical_parity_difference())