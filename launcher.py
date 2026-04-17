# launcher.py
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingOutput
from sagemaker.estimator import Estimator
import boto3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
ROLE = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::701604549449:role/service-role/AmazonSageMaker-ExecutionRole-20260415T144329")
IMAGE_URI = os.environ.get("IMAGE_URI", "701604549449.dkr.ecr.us-east-1.amazonaws.com/glm-ocr-pipeline:latest")
BUCKET = os.environ.get("BUCKET", "glm-ocr-gujarati-training-data")
PREFIX = "glm-ocr/gujarati-finetuning"

# Retrieve HF_TOKEN from .env or system environment
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Common environment variables for all jobs
ENV_VARS = {
    "HF_TOKEN": HF_TOKEN,
    "HF_HOME": "/opt/ml/code/huggingface_cache", # Use EBS space for cache
    "PYTHONUNBUFFERED": "1"
}

session = sagemaker.Session()

def run_data_prep():
    print("Step 1: Launching Data Preparation Job (Streaming -> Sharding)...")
    
    processor = ScriptProcessor(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        volume_size_in_gb=50,
        command=["/opt/ml/code/entrypoint.sh"],
        env={**ENV_VARS, "JOB_TYPE": "data_prep"}
    )
    
    processor.run(
        outputs=[
            ProcessingOutput(
                output_name="sharded_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{BUCKET}/{PREFIX}/shards"
            )
        ],
        arguments=["--dataset", "prthm29/gujarati_ocr_sharegpt"],
        wait=True
    )

def run_training(is_smoke_test=True):
    job_name = "glm-ocr-smoke" if is_smoke_test else "glm-ocr-full"
    print(f"Step 2: Launching {job_name}...")
    
    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_type="ml.g5.xlarge",
        instance_count=1,
        volume_size=100, # Fixed: Estimator uses 'volume_size'
        output_path=f"s3://{BUCKET}/{PREFIX}/output",
        base_job_name=job_name,
        environment={**ENV_VARS, "JOB_TYPE": "train"}
    )
    
    hyperparameters = {
        "smoke_test": str(is_smoke_test),
        "epochs": "1" if is_smoke_test else "3",
        "batch_size": "1",
        "learning_rate": "1e-4"
    }
    estimator.set_hyperparameters(**hyperparameters)
    
    # SageMaker will download the shards from S3 and mount them at /opt/ml/input/data/training
    estimator.fit({"training": f"s3://{BUCKET}/{PREFIX}/shards"}, wait=True)
    print(f"Training Complete. Model saved at: {estimator.model_data}")
    return estimator.model_data

if __name__ == "__main__":
    # run_data_prep()
    # model_data = run_training(is_smoke_test=True)
    pass