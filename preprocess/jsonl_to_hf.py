import argparse
import json
from datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()

with open(args.input_path, 'r') as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)
dataset.save_to_disk(args.output_path)
