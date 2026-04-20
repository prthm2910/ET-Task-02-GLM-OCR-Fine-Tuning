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
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_hyperparameters():
    """Load SageMaker hyperparameters from the local config file."""
    hp_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            try:
                return json.load(f)
            except Exception as e:
                logger.error(f"Error loading hyperparameters: {e}")
    return {}

def find_all_linear_names(model):
    """Identify all linear layers to target for LoRA fine-tuning."""
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            # Target the specific linear layer name
            lora_module_names.add(names[-1])
    
    # Exclude common output heads and vision model to focus on the language decoder
    for head in ["lm_head", "output_layer", "classifier"]:
        if head in lora_module_names:
            lora_module_names.remove(head)
            
    return list(lora_module_names)

def train():
    # 2. Load Hyperparameters
    sm_hps = load_hyperparameters()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--smoke_test", type=str, default=sm_hps.get("smoke_test", "False"))
    parser.add_argument("--epochs", type=int, default=int(sm_hps.get("epochs", 3)))
    parser.add_argument("--batch_size", type=int, default=int(sm_hps.get("batch_size", 1)))
    parser.add_argument("--learning_rate", type=float, default=float(sm_hps.get("learning_rate", 1e-4)))

    # Use parse_known_args to ignore SageMaker's internal "train" argument
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")
    
    is_smoke_test = str(args.smoke_test).lower() == "true"

    # 3. Load Dataset
    logger.info(f"Loading dataset from: {args.train_dir}")
    dataset = load_from_disk(args.train_dir)

    if is_smoke_test:
        logger.info("SMOKE TEST enabled: limiting to 20 samples.")
        dataset = dataset.select(range(min(20, len(dataset))))

    # 4. Initialize Model and Processor
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

    # 5. Apply LoRA
    target_modules = find_all_linear_names(model)
    logger.info(f"Targeting modules for LoRA: {target_modules}")

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 6. Preprocessing Logic (The Fix for 'labels' and 'loss')
    def preprocess_function(examples):
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_grid_thw = []
        
        for i in range(len(examples["messages"])):
            msg_list = examples["messages"][i]
            sample_images = examples["images"][i]
            if not isinstance(sample_images, list):
                sample_images = [sample_images]
            
            # Step A: Format for GLM-OCR
            formatted_messages = []
            for m in msg_list:
                role = m["role"]
                content_str = m["content"]
                if role == "user":
                    formatted_content = [
                        {"type": "image"}, 
                        {"type": "text", "text": content_str.replace("<image>", "").strip()}
                    ]
                else:
                    formatted_content = [{"type": "text", "text": content_str}]
                formatted_messages.append({"role": role, "content": formatted_content})

            # Step B: Render string and tokenize
            prompt = processor.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            inputs = processor(
                text=[prompt], 
                images=sample_images, 
                return_tensors="pt"
            )
            
            # Step C: Prepare Tensors and create labels
            input_ids = inputs["input_ids"].squeeze(0)
            
            # THE FIX: Create labels from input_ids
            # In causal training, labels are just a copy of the input_ids
            labels = input_ids.clone()
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_pixel_values.append(inputs["pixel_values"].squeeze(0))
            batch_image_grid_thw.append(inputs["image_grid_thw"].squeeze(0))
        
        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw
        }

    # Map the preprocessing
    train_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=args.batch_size,
        remove_columns=dataset.column_names
    )

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="no" if is_smoke_test else "epoch",
        remove_unused_columns=False, # CRITICAL: Keeps pixel_values and image_grid_thw
        report_to="none"
    )

    # 8. Use a Data Collator to handle multimodal padding
    data_collator = DataCollatorForSeq2Seq(
        processor.tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    logger.info("Ready. Starting Training...")
    trainer.train()
    
    logger.info("Training complete. Saving model to /opt/ml/model")
    trainer.save_model("/opt/ml/model")
    processor.save_pretrained("/opt/ml/model")

if __name__ == "__main__":
    train()
