# src/train.py
import argparse
import logging
import os
import json
import torch
from transformers import (
    AutoProcessor, 
    GlmOcrForConditionalGeneration, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_hyperparameters():
    hp_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            return json.load(f)
    return {}

def train():
    # 1. Load Hyperparameters from SageMaker JSON
    sm_hps = load_hyperparameters()

    parser = argparse.ArgumentParser()
    # SageMaker mounts S3 input data to /opt/ml/input/data/<channel_name>
    parser.add_argument("--train_dir", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--smoke_test", type=str, default=sm_hps.get("smoke_test", "False"))
    parser.add_argument("--epochs", type=int, default=int(sm_hps.get("epochs", 3)))
    parser.add_argument("--batch_size", type=int, default=int(sm_hps.get("batch_size", 1)))
    parser.add_argument("--learning_rate", type=float, default=float(sm_hps.get("learning_rate", 1e-4)))

    args = parser.parse_args()
    is_smoke_test = str(args.smoke_test).lower() == "true"

    logger.info(f"Loading sharded dataset from: {args.train_dir}")
    # load_from_disk automatically detects and merges the Arrow shards
    dataset = load_from_disk(args.train_dir)

    if is_smoke_test:
        logger.info("SMOKE TEST enabled: limiting to 20 samples.")
        dataset = dataset.select(range(20))

    # Initialize Processor & Model
    model_id = "zai-org/GLM-OCR"
    token = os.environ.get("HF_TOKEN")
    
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        token=token
    )

    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token
    )

    # LoRA Configuration - Targeting standard GLM Linear layers
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)


    # Preprocessing
    def preprocess_function(examples):
        batch_inputs = []
        for i in range(len(examples["messages"])):
            msg_list = examples["messages"][i]
            # Ensure images is a list of PIL objects
            sample_images = examples["images"][i]
            if not isinstance(sample_images, list):
                sample_images = [sample_images]
            
            # 1. Convert ShareGPT string content to Multimodal List format
            formatted_messages = []
            for m in msg_list:
                role = m["role"]
                content_str = m["content"]
                
                if role == "user":
                    # For GLM-OCR, the user prompt must contain the image type
                    # We replace the text placeholder <image> with the actual dict
                    formatted_content = [
                        {"type": "image"}, 
                        {"type": "text", "text": content_str.replace("<image>", "").strip()}
                    ]
                else:
                    formatted_content = [{"type": "text", "text": content_str}]
                
                formatted_messages.append({"role": role, "content": formatted_content})

            # 2. Apply chat template (handles both text and image preprocessing)
            inputs = processor.apply_chat_template(
                formatted_messages,
                images=sample_images,
                tokenize=True,
                add_generation_prompt=False, # We are training, so we include the assistant response
                return_dict=True,
                return_tensors="pt"
            )
            batch_inputs.append({k: v.squeeze(0) for k, v in inputs.items()})
        
        # Collate the results back into a dictionary of lists
        return {k: [dic[k] for dic in batch_inputs] for k in batch_inputs[0].keys()}

    # Map the preprocessing
    train_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=args.batch_size,
        remove_columns=dataset.column_names
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    logger.info("Starting Training...")
    trainer.train()
    
    logger.info("Saving Final LoRA Adapters...")
    trainer.save_model("/opt/ml/model")
    processor.save_pretrained("/opt/ml/model")

if __name__ == "__main__":
    train()