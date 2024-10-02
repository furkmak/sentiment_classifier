# run this first in terminal if needed
# mlflow server --host 127.0.0.1 --port 5000

import pandas as pd
import mlflow.sklearn
from mlflow import MlflowClient

client = MlflowClient()

# Load new data for prediction
new_data = pd.read_csv('./data/dataset_test.csv', encoding='utf-8')
X_new = new_data['text']

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Search for model versions
model_name = "model1"
model_versions = []

for mv in client.search_model_versions(f"name='{model_name}'"):
    # Convert mv to a dictionary and append it to the model_versions list
    model_versions.append(dict(mv))

# Find the latest model version (by version number or creation_timestamp)
latest_model_version = max(model_versions, key=lambda x: x['version'])  # or 'creation_timestamp'
print(latest_model_version)

# Extract the run_id of the latest model version
latest_run_id = latest_model_version['run_id']

model_uri = f"runs:/{latest_run_id}/sentiment_model"
pipe = mlflow.sklearn.load_model(model_uri)

# Make predictions
predictions = pipe.predict(X_new)

# Print or save predictions
print(predictions)

# Optionally, save predictions to a CSV file
new_data['predictions'] = predictions
new_data.to_csv('./data/test_pred.csv', index=False)
print("Predictions saved to 'test_pred.csv'")
