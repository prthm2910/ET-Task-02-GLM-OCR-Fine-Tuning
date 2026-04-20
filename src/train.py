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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk

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
    model.print_trainable_parameters()

    def preprocess_function(examples):
        batch_inputs = []
        for i in range(len(examples["messages"])):
            msg_list = examples["messages"][i]
            sample_images = examples["images"][i]
            if not isinstance(sample_images, list):
                sample_images = [sample_images]
            
            # Format message for GLM-OCR
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

            prompt = processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
            
            inputs = processor(text=[prompt], images=sample_images, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in inputs.items()}
            
            # THE FIX: Add labels for loss calculation
            # For autoregressive training, labels are the same as input_ids.
            item["labels"] = item["input_ids"].clone()
            batch_inputs.append(item)
            
        return {k: [d[k] for d in batch_inputs] for k in batch_inputs[0].keys()}

    train_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=args.batch_size,
        remove_columns=dataset.column_names
    )

    class MultimodalDataCollator:
        def __call__(self, features):
            # Extract labels before padding as processor.pad may ignore them
            labels = [f.pop("labels") for f in features] if "labels" in features[0] else None
            
            # Pad the remaining features (input_ids, attention_mask, pixel_values, etc.)
            batch = processor.pad(features, return_tensors="pt")
            
            if labels is not None:
                max_label_length = max(len(l) for l in labels)
                padded_labels = []
                for l in labels:
                    # Pad labels with -100 to ignore padding in loss calculation
                    padding_length = max_label_length - len(l)
                    if padding_length > 0:
                        padded_labels.append(torch.cat([l, torch.full((padding_length,), -100, dtype=torch.long)]))
                    else:
                        padded_labels.append(l)
                batch["labels"] = torch.stack(padded_labels)
                
            # SAFETY CHECK: Ensure mm_token_type_ids exists if needed by the model
            if "mm_token_type_ids" not in batch and "input_ids" in batch:
                batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
                
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
        remove_unused_columns=False, # Crucial for multimodal
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MultimodalDataCollator()
    )

    logger.info("Starting Training...")
    trainer.train()
    
    logger.info("Saving artifacts...")
    trainer.save_model("/opt/ml/model")
    processor.save_pretrained("/opt/ml/model")

if __name__ == "__main__":
    train()
