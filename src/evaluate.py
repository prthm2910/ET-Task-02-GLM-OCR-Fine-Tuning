# src/evaluate.py
import argparse
import json
import logging
import torch
from io import BytesIO
import boto3
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj['Body'].read()

def get_image_from_s3(bucket, image_key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=image_key)
    return Image.open(BytesIO(obj['Body'].read())).convert("RGB")

import argparse
import json
import logging
import torch
import os
import tarfile
from io import BytesIO
import boto3
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="glm-ocr/processed_data")
    parser.add_argument("--num_samples", type=int, default=10)
    
    args = parser.parse_args()
    
    # 1. Prepare Model Path (Handle SageMaker model.tar.gz)
    input_dir = "/opt/ml/input/data/model"
    adapter_path = input_dir
    
    # Check if model.tar.gz exists (Standard SageMaker Output)
    tar_path = os.path.join(input_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        logger.info(f"Extracting model artifacts from {tar_path}...")
        extract_path = "/tmp/model_extracted"
        os.makedirs(extract_path, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        adapter_path = extract_path
    
    # 2. Load Data
    logger.info(f"Loading metadata from S3: {args.bucket}/{args.prefix}/data.json")
    # Note: Using the shards directly for evaluation
    from datasets import load_from_disk
    # The training shards are mounted at /opt/ml/input/data/training usually, 
    # but here we access the bucket directly for evaluation samples to keep it simple.
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=args.bucket, Key=f"{args.prefix}/data.json")
    raw_data = json.loads(obj['Body'].read().decode('utf-8'))
    eval_samples = raw_data[-args.num_samples:]
    
    # 3. Load Model & Processor
    model_id = "zai-org/GLM-OCR"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load Trained Adapters
    logger.info(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    results = []
    
    logger.info("Starting Inference on Evaluation samples...")
    for i, item in enumerate(eval_samples):
        messages = item['messages']
        images = [get_image_from_s3(args.bucket, k) for k in item['images']]
        
        # Prepare input
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=images, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )
        
        # Decode
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        ground_truth = next(m['content'] for m in messages if m['role'] == 'assistant')
        
        logger.info(f"Sample {i+1}:")
        logger.info(f"GT: {ground_truth}")
        logger.info(f"AI: {response}")
        
        results.append({
            "sample_id": i,
            "ground_truth": ground_truth,
            "prediction": response
        })

    # 4. Save results to S3
    output_key = f"{args.prefix}/evaluation_results.json"
    s3 = boto3.client('s3')
    s3.put_object(Bucket=args.bucket, Key=output_key, Body=json.dumps(results, ensure_ascii=False))
    logger.info(f"Evaluation complete. Results saved to s3://{args.bucket}/{output_key}")

if __name__ == "__main__":
    evaluate()