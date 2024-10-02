import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric
from sentence_transformers import SentenceTransformer
import os

# Set the current directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Load current data
current = pd.read_csv('data/dataset_test.csv', encoding='utf-8')
current_data = current.sample(n=1000)

# Load reference data
reference = pd.read_csv('data/dataset.csv', encoding='utf-8')
reference_data = reference.sample(n=1000)

# Select relevant columns
reference_data = reference_data[['text']]
current_data = current_data[['text']]

# Use sentence transformer to convert the text into embeddings
model_miniLM = SentenceTransformer('all-MiniLM-L6-v2')

# Convert the text columns into embeddings
reference_embeddings = model_miniLM.encode(reference_data['text'].tolist())
current_embeddings = model_miniLM.encode(current_data['text'].tolist())

# Convert embeddings into DataFrame
reference_df = pd.DataFrame(reference_embeddings, columns=[f'col_{i}' for i in range(reference_embeddings.shape[1])])
current_df = pd.DataFrame(current_embeddings, columns=[f'col_{i}' for i in range(current_embeddings.shape[1])])

# Create a ColumnMapping to define embedding columns
column_mapping = ColumnMapping(embeddings={'small_subset': reference_df.columns})

# Initialize Evidently report for embeddings drift
report = Report(metrics=[
    EmbeddingsDriftMetric('small_subset')
])

# Run the report
report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

# Save the report as an HTML file
report.save_html('reports/data_drift_report.html')

# Extract the drift score
drift_metric = report.as_dict()['metrics'][0]['result']['drift_score']

# Set a threshold for drift
drift_threshold = 0.7

# Check if drift exceeds the threshold
if drift_metric > drift_threshold:
    print(f"ALERT: Drift score of {drift_metric} exceeds threshold of {drift_threshold}!")
else:
    print(f"Drift score is {drift_metric}, within acceptable range.")

print("Embeddings drift report generated successfully.")
