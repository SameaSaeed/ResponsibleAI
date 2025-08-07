import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import shap

# Load the Adult dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

data = pd.read_csv(url, names=columns, sep=',\s', engine='python')

# Preprocessing: Encode categorical features
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data into features and target
X = data.drop('income', axis=1)
y = data['income']

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Create a SHAP explainer object
explainer = shap.KernelExplainer(model.predict, X_train[:100])  # Use a subset of the training data

# Get SHAP values for a sample of predictions
shap_values = explainer.shap_values(X_test[:10])

# Visualize the SHAP values for the first prediction
shap.initjs()  # Initialize JavaScript visualization
shap.force_plot(shap_values[0], X_test[:10][0])

shap.summary_plot(shap_values[0], X_test[:10])

# Evaluate the fairness across different gender groups (e.g., male vs female)
male_data = X_test[y_test == 0]  # Filter male data
female_data = X_test[y_test == 1]  # Filter female data

# Get SHAP values for males and females
male_shap_values = explainer.shap_values(male_data)
female_shap_values = explainer.shap_values(female_data)

# Compare the feature importance across gender
shap.summary_plot(male_shap_values[0], male_data)
shap.summary_plot(female_shap_values[0], female_data)