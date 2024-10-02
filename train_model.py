# run this first in terminal
# mlflow server --host 127.0.0.1 --port 5000

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib

# Load your dataset
data = pd.read_csv('data/dataset.csv')

# Assume the text is in 'text' and the target in 'label'
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_params = {
    "n_estimators": 100,
    "random_state": 42
}

proc_params = {
    "max_features": 5000,
    "stop_words": 'english'
}

# Create a pipeline that includes the TfidfVectorizer and RandomForestClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(**proc_params)),
    ('rf', RandomForestClassifier(**model_params))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# Print metrics
print(f'Model accuracy: {accuracy * 100:.2f}%')
print(f'Model precision: {precision * 100:.2f}%')
print(f'Model recall: {recall * 100:.2f}%')
print(f'Model F1-score: {f1 * 100:.2f}%')
print(f'Confusion Matrix:\n{cm}')


# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create or set the experiment in MLflow
experiment_name = "sentiment_classifier"
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run():

    # Log the number of features created by TfidfVectorizer
    mlflow.log_param('tfidf_max_features', 5000)
    mlflow.log_param('tfidf_stop_words', 'english')

    # Initialize RandomForestClassifier
    n_estimators = 100
    random_state = 42

    # Log model parameters
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('random_state', random_state)

    # Log metrics to MLflow
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic RF model for sentiment classification")

    # Infer the signature (input/output schema) from the model predictions
    signature = infer_signature(X_train, pipeline.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="sentiment_model",
        signature=signature,
        #input_example=X_train,
        registered_model_name="model1"
    )

    print("Model and metrics logged to MLflow.")

# Save the model and vectorizer locally as well
joblib.dump(pipeline, 'model/rf_pipe.pkl')
print("Model and vectorizer saved locally.")
