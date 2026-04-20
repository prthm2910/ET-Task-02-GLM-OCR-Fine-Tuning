# launcher.py
import sagemaker
from sagemaker.processing import Processor, ProcessingOutput
from sagemaker.estimator import Estimator
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
ROLE = os.environ.get("SAGEMAKER_ROLE", "")
IMAGE_URI = os.environ.get("IMAGE_URI", "")
BUCKET = os.environ.get("BUCKET", "")
PREFIX = "glm-ocr/gujarati-finetuning"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ENV_VARS = {
    "HF_TOKEN": HF_TOKEN,
    "HF_HOME": "/opt/ml/code/huggingface_cache",
    "PYTHONUNBUFFERED": "1"
}

def run_parallel_data_prep():
    splits = ["train", "test", "val"]
    print(f"Launching {len(splits)} parallel jobs for: {splits}")
    
    for split in splits:
        print(f"Creating job for split: {split}...")
        processor = Processor(
            image_uri=IMAGE_URI,
            role=ROLE,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            volume_size_in_gb=50,
            base_job_name=f"prep-gujarati-{split}",
            env={**ENV_VARS, "JOB_TYPE": "data_prep"}
        )
        
        # wait=False allows us to launch the next one immediately
        # logs=False is required when wait=False to avoid ValueError
        processor.run(
            outputs=[
                ProcessingOutput(
                    output_name="sharded_data",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{BUCKET}/{PREFIX}/shards/{split}"
                )
            ],
            arguments=["--split", split],
            wait=False,
            logs=False
        )
    print("All jobs launched! Check the SageMaker Console to monitor them.")

def run_training(is_smoke_test=True):
    # Important: Point specifically to the 'train' folder created by the prep job
    train_input = f"s3://{BUCKET}/{PREFIX}/shards/train"
    
    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_type="ml.g5.xlarge",
        instance_count=1,
        volume_size=100,
        output_path=f"s3://{BUCKET}/{PREFIX}/output",
        base_job_name="glm-ocr-training",
        environment={**ENV_VARS, "JOB_TYPE": "train"}
    )
    
    estimator.set_hyperparameters(smoke_test=str(is_smoke_test))
    estimator.fit({"training": train_input}, wait=True)

if __name__ == "__main__":
    # run_parallel_data_prep()
    model_data = run_training(is_smoke_test=True)
