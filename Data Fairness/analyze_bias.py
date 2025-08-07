import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.metrics import accuracy_score

# Sample dataset: Gender bias in hiring (mocked)
data = pd.DataFrame({
    "education": [1, 2, 3, 4, 2, 3, 4, 1, 2, 3],
    "experience": [1, 3, 5, 7, 2, 6, 8, 1, 2, 5],
    "gender": ["male", "female", "female", "male", "female", "female", "male", "male", "female", "male"],
    "hired":   [0, 1, 1, 1, 0, 1, 1, 0, 0, 1]
})

# Encode gender to numeric
data["gender_encoded"] = data["gender"].map({"male": 0, "female": 1})

X = data[["education", "experience"]]
y = data["hired"]
sensitive_feature = data["gender_encoded"]

# Split dataset
X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate fairness
acc = accuracy_score(y_test, predictions)
metrics = MetricFrame(
    metrics={"selection_rate": selection_rate},
    y_true=y_test,
    y_pred=predictions,
    sensitive_features=sf_test
)

print("\nOverall Accuracy:", acc)
print("\nSelection Rate by Gender:\n", metrics.by_group)
print("\nDemographic Parity Difference:",
      demographic_parity_difference(y_test, predictions, sensitive_features=sf_test))