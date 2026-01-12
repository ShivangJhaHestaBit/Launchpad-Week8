import json
import os
import random
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm

RAW_FILE = "src/data/qa.en.go.json"
OUTPUT_DIR = "src/data"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"

MAX_SAMPLES = 1000
TRAIN_RATIO = 0.9

MODEL_NAME = "Qwen/Qwen2-7B"
MAX_TOTAL_TOKENS = 1200
MIN_TOTAL_TOKENS = 50

EXTRACTION_RATIO = 0.2

random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "stats"), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

raw_data = []
with open(RAW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        raw_data.append(json.loads(line))

raw_data = raw_data[:MAX_SAMPLES]
print(f"Loaded {len(raw_data)} raw samples")

def extract_code_blocks(text):
    fenced = re.findall(r"```.*?```", text, re.DOTALL)
    indented = re.findall(r"(?:\n {4}.+)+", text)
    return fenced + indented

def extract_go_functions(code):
    return re.findall(r'func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', code)

processed = []

for item in tqdm(raw_data):
    base_instruction = (
        "You are an expert Go language developer. "
        "Answer the following question clearly and accurately."
    )

    input_text = f"Title: {item['title']}\n\nQuestion:\n{item['question'].strip()}"
    output_text = item["answer"].strip()

    if not input_text or not output_text:
        continue

    processed.append({
        "instruction": base_instruction,
        "input": input_text,
        "output": output_text
    })

    if random.random() < EXTRACTION_RATIO:
        code_blocks = extract_code_blocks(output_text)
        for code in code_blocks:
            go_funcs = extract_go_functions(code)
            if go_funcs:
                processed.append({
                    "instruction": "Extract all Go function names defined in the following code.",
                    "input": code,
                    "output": ", ".join(sorted(set(go_funcs)))
                })

print(f"Total samples before filtering: {len(processed)}")

token_lengths = []
cleaned = []

for sample in processed:
    total_text = sample["instruction"] + sample["input"] + sample["output"]
    token_len = len(tokenizer(total_text, truncation=False)["input_ids"])

    token_lengths.append(token_len)

    if MIN_TOTAL_TOKENS <= token_len <= MAX_TOTAL_TOKENS:
        cleaned.append(sample)

print(f"After outlier removal: {len(cleaned)} samples")

plt.figure(figsize=(8, 5))
plt.hist(token_lengths, bins=50)
plt.title("Total Token Length Distribution")
plt.xlabel("Tokens")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_DIR, "stats/token_hist.png"))
plt.close()

qa_lengths = [
    len(tokenizer(s["input"] + s["output"])["input_ids"])
    for s in cleaned
]

plt.figure(figsize=(8, 5))
plt.hist(qa_lengths, bins=50)
plt.title("QA Token Length Distribution")
plt.xlabel("Tokens")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_DIR, "stats/qa_length_dist.png"))
plt.close()

random.shuffle(cleaned)
split_idx = int(len(cleaned) * TRAIN_RATIO)

train_data = cleaned[:split_idx]
val_data = cleaned[split_idx:]

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(os.path.join(OUTPUT_DIR, TRAIN_FILE), train_data)
write_jsonl(os.path.join(OUTPUT_DIR, VAL_FILE), val_data)

print("Dataset ready for QA + Reasoning + Extraction")
