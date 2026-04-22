# src/train.py
import argparse
import logging
import os
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoProcessor, 
    GlmOcrForConditionalGeneration, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_hyperparameters():
    hp_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hp_path):
        with open(hp_path, "r") as f:
            try:
                return json.load(f)
            except Exception as e:
                logger.error(f"Error loading hyperparameters: {e}")
    return {}

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[-1])
    for head in ["lm_head", "output_layer", "classifier"]:
        if head in lora_module_names:
            lora_module_names.remove(head)
    return list(lora_module_names)

def train():
    sm_hps = load_hyperparameters()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--smoke_test", type=str, default=sm_hps.get("smoke_test", "False"))
    parser.add_argument("--epochs", type=int, default=int(sm_hps.get("epochs", 3)))
    parser.add_argument("--batch_size", type=int, default=int(sm_hps.get("batch_size", 1)))
    parser.add_argument("--learning_rate", type=float, default=float(sm_hps.get("learning_rate", 1e-4)))

    args, unknown = parser.parse_known_args()
    is_smoke_test = str(args.smoke_test).lower() == "true"

    dataset = load_from_disk(args.train_dir)
    if is_smoke_test:
        dataset = dataset.select(range(min(20, len(dataset))))

    model_id = "zai-org/GLM-OCR"
    token = os.environ.get("HF_TOKEN")
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token
    )

    target_modules = find_all_linear_names(model)
    config = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, config)

    def preprocess_function(examples):
        processed_batches = []
        for i in range(len(examples["messages"])):
            msg_list = examples["messages"][i]
            sample_images = examples["images"][i]
            if not isinstance(sample_images, list):
                sample_images = [sample_images]
            
            prompt = processor.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[prompt], images=sample_images, return_tensors="pt")
            
            item = {k: v.squeeze(0) for k, v in inputs.items()}
            item["labels"] = item["input_ids"].clone()
            processed_batches.append(item)
            
        return {k: [d[k] for d in processed_batches] for k in processed_batches[0].keys()}

    train_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=args.batch_size,
        remove_columns=dataset.column_names
    )

    class ManualMultimodalDataCollator:
        def __call__(self, features):
            batch = {}
            
            # 1. Manually pad token-based sequences
            input_ids = [f["input_ids"] for f in features]
            labels = [f["labels"] for f in features]
            mm_ids = [f["mm_token_type_ids"] for f in features]
            masks = [f["attention_mask"] for f in features]
            
            batch["input_ids"] = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
            batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)
            batch["mm_token_type_ids"] = pad_sequence(mm_ids, batch_first=True, padding_value=0)
            batch["attention_mask"] = pad_sequence(masks, batch_first=True, padding_value=0)
            
            # Handle rope_deltas if present
            if "rope_deltas" in features[0]:
                rope_deltas = [f["rope_deltas"] for f in features]
                batch["rope_deltas"] = pad_sequence(rope_deltas, batch_first=True, padding_value=0)
            
            # 2. Concatenate multimodal tensors on dim=0
            # Mapping 'pixel_values' to 'images' for model forward pass compatibility
            if "pixel_values" in features[0]:
                batch["images"] = torch.cat([f["pixel_values"] for f in features], dim=0)
            
            if "image_grid_thw" in features[0]:
                batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)
                
            return batch

    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="no" if is_smoke_test else "epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=ManualMultimodalDataCollator()
    )

    logger.info("Starting Training...")
    trainer.train()
    
    logger.info("Saving artifacts...")
    trainer.save_model("/opt/ml/model")
    processor.save_pretrained("/opt/ml/model")

if __name__ == "__main__":
    train()
