# src/data_prep.py
import argparse
import logging
import os
from datasets import load_dataset, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data(dataset_name, output_dir):
    logger.info(f"Connecting to Hugging Face dataset: {dataset_name} in streaming mode.")
    
    # 1. Stream the dataset using the token from environment
    token = os.environ.get("HF_TOKEN")
    stream_ds = load_dataset(dataset_name, split="train", streaming=True, token=token)
    
    # 2. Use a generator to materialize the stream
    def gen():
        for item in stream_ds:
            yield item
            
    logger.info("Materializing stream and saving to sharded Arrow format...")
    # Dataset.from_generator is memory efficient
    materialized_dataset = Dataset.from_generator(gen)
    
    # 3. save_to_disk automatically creates shards
    # We set max_shard_size to keep files large and S3 transfers fast
    materialized_dataset.save_to_disk(output_dir, max_shard_size="500MB")
    
    logger.info(f"Dataset sharded and saved locally to {output_dir}. SageMaker will now upload to S3.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="prthm29/gujarati_ocr_sharegpt")
    # /opt/ml/processing/output is the standard SageMaker path for auto-S3 upload
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    
    args = parser.parse_args()
    prepare_data(args.dataset, args.output_dir)
