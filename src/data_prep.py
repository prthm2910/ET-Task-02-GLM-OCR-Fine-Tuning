# src/data_prep.py
import argparse
import logging
import os
from datasets import load_dataset, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data(dataset_name, split_name, output_dir):
    logger.info(f"Connecting to Hugging Face dataset: {dataset_name} (Split: {split_name})")
    
    token = os.environ.get("HF_TOKEN")
    # Use the provided split_name instead of hardcoded "train"
    stream_ds = load_dataset(dataset_name, split=split_name, streaming=True, token=token)
    
    def gen():
        for item in stream_ds:
            yield item
            
    logger.info(f"Materializing {split_name} stream and saving to {output_dir}...")
    materialized_dataset = Dataset.from_generator(gen)
    materialized_dataset.save_to_disk(output_dir, max_shard_size="500MB")
    
    logger.info(f"Split {split_name} sharded and saved to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="prthm29/gujarati_ocr_sharegpt")
    parser.add_argument("--split", type=str, default="train") # Added this
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    
    args = parser.parse_args()
    prepare_data(args.dataset, args.split, args.output_dir)
