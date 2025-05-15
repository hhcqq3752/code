import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from utils import RewardDataCollatorWithPadding, RewardTrainer, build_dataset

@dataclass
class Arguments:
    # Data
    train_set_path: Optional[str] = field(default="")
    max_length: Optional[int] = field(default=16384)

    # Model
    model_name: Optional[str] = field(default="Llama-3.1-8B-Instruct")

    # Training
    per_device_train_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=1e-6)
    weight_decay: Optional[float] = field(default=0.001)
    num_train_epochs: Optional[int] = field(default=1)
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: Optional[str] = field(default="linear")
    warmup_ratio: Optional[float] = field(default=0.03)
    warmup_steps: Optional[int] = field(default=0)
    max_steps: Optional[int] = field(default=-1)

    # Other
    output_dir: Optional[str] = field(default="reward_model")
    save_steps: Optional[int] = field(default=999999)
    run_name: Optional[str] = field(default="reward_model")
    logging_steps: Optional[int] = field(default=1)
    save_total_limit: Optional[int] = field(default=999999)
    seed: Optional[int] = field(default=42)
    data_seed: Optional[int] = field(default=None)
    resume_from_checkpoint: Optional[bool] = field(default=False)
    use_liger_kernel: Optional[bool] = field(default=False)



args = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
os.makedirs(args.output_dir, exist_ok=True)
args.resume_from_checkpoint = None if not args.resume_from_checkpoint else args.resume_from_checkpoint
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
if "qwen" in args.model_name.lower():
    tokenizer.pad_token = "<|vision_pad|>"
elif "llama" in args.model_name.lower():
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
else:
    raise ValueError(f"Unsupported model: {args.model_name}")
tokenizer.truncation_side = "left"
tokenizer.model_max_length = args.max_length
train_dataset = build_dataset(args.train_set_path, args.seed, tokenizer=None)
print("Training set:", len(train_dataset))
print("Training set example:", train_dataset[0])

# Define the trainer
training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    save_strategy="steps",
    save_steps=args.save_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    remove_unused_columns=False,
    bf16=True,
    logging_strategy="steps",
    logging_steps=args.logging_steps,
    optim=args.optim,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    warmup_steps=args.warmup_steps,
    max_steps=args.max_steps,
    report_to="none",
    run_name=args.run_name,
    save_total_limit=args.save_total_limit,
    seed=args.seed,
    data_seed=args.data_seed,
    use_liger_kernel=args.use_liger_kernel,
)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = not args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=args.max_length
    ),
)
trainer.train(resume_from_checkpoint=True if len(os.listdir(args.output_dir)) > 0 else None)
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
with open(os.path.join(args.output_dir, "args.json"), "w") as f:
    json.dump(args.__dict__, f, indent=4)
