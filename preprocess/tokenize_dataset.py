import os
from datasets import load_from_disk
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if "Llama" in args.model_name:
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
elif "Qwen" in args.model_name:
    tokenizer.pad_token = "<|vision_pad|>"
tokenizer.truncation_side = "left"
tokenizer.model_max_length = 16384

def build_dataset(train_path, tokenizer=None):
    def tokenize_func(sample):
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        )
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        )

        if tokenizer.bos_token and tokenizer.bos_token in sample["positive"]:
            sample["positive"] = sample["positive"][len(tokenizer.bos_token):]
        if tokenizer.bos_token and tokenizer.bos_token in sample["negative"]:
            sample["negative"] = sample["negative"][len(tokenizer.bos_token):]

        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample

    dataset = load_from_disk(train_path)
    if tokenizer is not None:
        dataset = dataset.map(tokenize_func, num_proc=os.cpu_count())

    dataset = dataset.select_columns(
        ["input_ids_j", "attention_mask_j", "input_ids_k", "attention_mask_k"]
    )
    return dataset

# Use multi-processing via the map method (adjust num_proc as needed)
ds_tokens = build_dataset(args.dataset_path, tokenizer)
print(ds_tokens[0])

# Save the processed dataset
output_path = args.dataset_path + "_tokenized"
ds_tokens.save_to_disk(output_path)
