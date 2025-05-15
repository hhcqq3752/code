import sys
import re
import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import random

from prompts import ATTRIBUTE_LABELING_PROMPT, ANNOTATION_PROMPT

import json
import re
import ast
from typing import Optional, Dict

import re
import json

def parse_attribute_labeling_output(text: str) -> dict | None:
    """
    Extracts and parses the first valid JSON object from a string that may contain extra text.
    Includes validation for required keys and allowed values for task_category, preference_objectivity, and controversiality.

    Args:
        text (str): A string potentially containing a JSON object.

    Returns:
        dict | None: Parsed JSON object if valid and complete, otherwise None.
    """
    # Allowed values from the prompt
    valid_task_categories = {
        "Advice seeking", "Brainstorming", "Coding & Debugging", "Creative writing",
        "Data analysis", "Editing", "Information seeking", "Math",
        "Planning", "Reasoning", "Role playing", "Other"
    }
    valid_preference_objectivities = {"Objective", "Subjective"}
    valid_controversialities = {"Low", "Medium", "High"}

    # Regular expression to match a JSON object
    json_pattern = re.compile(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', re.DOTALL)

    # Find all JSON-like substrings
    matches = json_pattern.findall(text)

    for match in matches:
        try:
            parsed = json.loads(match)
            # Required keys
            required_keys = {
                "task_category",
                "preference_objectivity",
                "controversiality",
                "desired_attributes",
                "annotation_guideline"
            }
            # Basic key check
            if not isinstance(parsed, dict) or not required_keys.issubset(parsed.keys()):
                continue
            # Value checks
            if parsed["task_category"] not in valid_task_categories:
                continue
            if parsed["preference_objectivity"] not in valid_preference_objectivities:
                continue
            if parsed["controversiality"] not in valid_controversialities:
                continue
            return parsed
        except json.JSONDecodeError:
            continue

    return None



parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="meta-llama/Llama-3.3-70B-Instruct-NoSys",
)
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--n", type=int, default=64)
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--max_tokens", type=int, default=16384)
parser.add_argument("--num_workers", type=int, default=256)
parser.add_argument("--dp", type=int, default=4)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--max_passes", type=int, default=10)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--task", type=str, default="attribute_labeling", choices=["attribute_labeling", "annotation"])
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

# Prepare base URLs and tokenizer.
base_urls = [f"http://127.0.0.1:{8000 + i}/v1" for i in range(args.dp)]
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# Load the entire input dataset and record all ids.
with open(args.input_path, "r", encoding="utf-8") as f:
    all_dataset = [json.loads(line) for line in f]
if args.debug:
    all_dataset = all_dataset[:100]
all_ids = set([item["id"] for item in all_dataset])
assert len(all_ids) == len(all_dataset), f"len(all_ids) = {len(all_ids)}, len(all_dataset) = {len(all_dataset)}"

# If output exists and overwrite is specified, remove it; otherwise, leave it intact.
if args.overwrite and os.path.exists(args.output_path):
    print(f"Overwriting {args.output_path}")
    os.remove(args.output_path)
else:
    print(f"Appending to {args.output_path}")


def format_conversation(conv):
    role_mapping = {"user": "User", "assistant": "Assistant"}
    formatted_conv = ""
    for msg in conv:
        if msg['role'] == "human":
            msg['role'] = "user"
        formatted_conv += f"{role_mapping[msg['role']]}: {msg['content']}\n"
    return formatted_conv.strip()


def format_prompt_for_attribute_labeling(x):
    conv = x["chosen"][:-1]
    good_response = f"Assistant: {x['chosen'][-1]['content']}"
    bad_response = f"Assistant: {x['rejected'][-1]['content']}"
    conv = format_conversation(conv)
    # Randomly swap the good and bad responses.
    if random.random() < 0.5:
        good_response, bad_response = bad_response, good_response
    else:
        good_response, bad_response = good_response, bad_response
    prompt = ATTRIBUTE_LABELING_PROMPT.format(conversation_history=conv, candidate_1=good_response, candidate_2=bad_response)
    return prompt


def format_single_conv(x):
    conv = x["chosen"][:-1]
    good_response = f"Assistant: {x['chosen'][-1]['content']}"
    bad_response = f"Assistant: {x['rejected'][-1]['content']}"
    conv = format_conversation(conv)
    flipped = False
    if random.random() < 0.5:
        good_response, bad_response = bad_response, good_response
        flipped = True
    else:
        good_response, bad_response = good_response, bad_response
    preferred = "Candidate 2" if flipped else "Candidate 1"
    template = "[START OF CONVERSATION HISTORY]\n{conversation_history}\n[END OF CONVERSATION HISTORY]\n[START OF CANDIDATE 1 RESPONSE]\n{candidate_1}\n[END OF CANDIDATE 1 RESPONSE]\n[START OF CANDIDATE 2 RESPONSE]\n{candidate_2}\n[END OF CANDIDATE 2 RESPONSE]"
    template += f"```json\n{{'preferred': '{preferred}'}}\n```"
    return template.format(conversation_history=conv, candidate_1=good_response, candidate_2=bad_response)


def format_prompt_for_annotation(human_examples, x):
    human_examples = "\n\n".join([format_single_conv(x) for x in human_examples])
    conv = x["chosen"][:-1]
    good_response = f"Assistant: {x['chosen'][-1]['content']}"
    bad_response = f"Assistant: {x['rejected'][-1]['content']}"
    conv = format_conversation(conv)
    if random.random() < 0.5:
        good_response, bad_response = bad_response, good_response
    else:
        good_response, bad_response = good_response, bad_response
    prompt = ANNOTATION_PROMPT.format(conversation_history=conv, candidate_1=good_response, candidate_2=bad_response, human_examples=human_examples)
    return prompt


def process_query(x, base_url):
    client = OpenAI(api_key="vllm", base_url=base_url, timeout=3600)
    if args.task == "attribute_labeling":
        prompt = format_prompt_for_attribute_labeling(x)
    elif args.task == "annotation":
        prompt = format_prompt_for_annotation(x)
    x["prompt"] = prompt
    prompt = [{"role": "user", "content": prompt}]
    # try:
    prompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    if tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
        prompt = prompt[len(tokenizer.bos_token) :]

    num_prompt_tokens = tokenizer.encode(prompt)
    num_max_tokens = args.max_tokens - len(num_prompt_tokens) - 1

    response = client.completions.create(
        n=1,
        model=args.model_name_or_path,
        prompt=[prompt] * args.n,
        max_tokens=num_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    # Collect all responses and remove duplicates
    responses = [choice.text for choice in response.choices]
    x["responses"] = responses
    if args.task == "attribute_labeling":
        x["attributes"] = [parse_attribute_labeling_output(r) for r in responses]
    elif args.task == "annotation":
        x["annotation"] = responses
    return x
    # except Exception as e:
    #     print(f"Error processing query {x['id']}: {e}")
    #     x["responses"] = []
    #     return x


def deduplicate_output():
    """
    Read the output file, deduplicate by id and rewrite the file.
    Returns the set of processed ids.
    """
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8", errors="ignore") as f:
            # processed_data = [json.loads(line) for line in f]
            processed_data = []
            for line in f:
                if not line.strip():
                    continue
                try:
                    processed_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Ignoring line {line} due to JSONDecodeError")
                    continue
                except Exception as e:
                    print(f"Ignoring line {line} due to {e}")
                    continue
        processed_ids = set()
        unique_data = []
        for data in processed_data:
            if data["id"] not in processed_ids:
                processed_ids.add(data["id"])
                unique_data.append(data)
        # Write back deduplicated data.
        with open(args.output_path, "w", encoding="utf-8") as f:
            for data in unique_data:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return processed_ids
    else:
        return set()


pass_num = 1
# Get the current processed ids from any existing output.
processed_ids = deduplicate_output()
missing_ids = all_ids - processed_ids

# Continue processing passes until there are no missing ids.
while missing_ids:
    print(f"\n=== Pass {pass_num}: Processing {len(missing_ids)} missing queries ===")
    # Filter the dataset for samples whose id is missing.
    current_dataset = [
        x for x in all_dataset if x["id"] in missing_ids
    ]

    # Process the current batch of queries using a ThreadPoolExecutor.
    with open(args.output_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_query, x, base_urls[i % len(base_urls)])
                for i, x in enumerate(current_dataset)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing queries (Pass {pass_num})",
            ):
                try:
                    result = future.result()
                except Exception as e:
                    # print and immediately exit
                    print(f"Fatal error processing query: {e}", file=sys.stderr)
                    sys.exit(1)
                # Only write the sample if at least one response was generated.
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
    # Deduplicate the output file and update the set of processed ids.
    processed_ids = deduplicate_output()
    missing_ids = all_ids - processed_ids
    print(f"Pass {pass_num} complete. {len(missing_ids)} queries still missing.")
    pass_num += 1
    if pass_num >= args.max_passes:
        break

if len(missing_ids) == 0:
    print("\nAll queries processed!")
else:
    print(f"\n{len(missing_ids)} queries still missing after {args.max_passes} passes.")
