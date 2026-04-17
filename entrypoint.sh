#!/bin/bash
# entrypoint.sh

echo "Starting SageMaker Job Type: $JOB_TYPE"

if [ "$JOB_TYPE" == "data_prep" ]; then
    echo "Routing to Data Preparation Script..."
    python3 src/data_prep.py "$@"

elif [ "$JOB_TYPE" == "train" ]; then
    echo "Routing to Training Script..."
    python3 src/train.py "$@"

elif [ "$JOB_TYPE" == "evaluate" ]; then
    echo "Routing to Evaluation Script..."
    python3 src/evaluate.py "$@"

else
    echo "Error: Unknown or missing JOB_TYPE environment variable."
    echo "Please set JOB_TYPE to 'data_prep', 'train', or 'evaluate'."
    exit 1
fi