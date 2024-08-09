#!/bin/bash

# Preprocess data
PREPROCESSED_DATA=$(curl -s -X POST http://localhost:5000/preprocess)
echo "Preprocessed Data: $PREPROCESSED_DATA"

# Train model
TRAIN_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "$PREPROCESSED_DATA" http://localhost:5000/train)
echo "Train Response: $TRAIN_RESPONSE"

# Prepare data for inference using Python
INFER_DATA=$(python3 -c "
import json
import pandas as pd

# Load and process the preprocessed data
data = json.loads('$PREPROCESSED_DATA')

# Convert to DataFrame
df = pd.DataFrame(data)

# Select the first 5 rows and remove the 'target' column
infer_data = df.head(5).drop(columns=['target']).to_json(orient='records')
print(infer_data)
")

echo "Inference Data: $INFER_DATA"

# Perform inference
INFER_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "$INFER_DATA" http://localhost:5000/infer)
echo "Inference Response: $INFER_RESPONSE"

# Keep the terminal open
read -p "Press Enter to close..."
