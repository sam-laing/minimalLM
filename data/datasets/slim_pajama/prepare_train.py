"""
Optimized SlimPajama preprocessing script.
Fixes:
1. Keeps data streaming directly into .map() via generator (bypasses heavy network disk write).
2. Sets node-local temporary caching (/tmp).
3. Uses dynamic Slurm core counts for num_proc.
4. Uses lightweight batch size for chunk concatenation to avoid Python RAM bottleneck.
"""

import os
import tempfile
from itertools import chain
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# --------------------------------------------------------------------
# Config & Paths

out_path = os.environ.get("PLAINLM_DATA_OUT", "/workspace/data/lm/sp_1.8B_tokens")
dataset_name = "gmongaras/SlimPajama-627B_Reupload"
split = 'train'

nrows = 1_800_000 # ~ 1.8B tokens
seq_len = 2048
max_seq_length = seq_len + 1
ordering = "randomized"

# Override with PLAINLM_NUM_CPUS if the shared node isn't fully yours,
# otherwise default to all visible cores (fine on a dedicated RunPod pod).
num_cpus = int(os.environ.get("PLAINLM_NUM_CPUS", os.cpu_count() or 8))

# Force HF Dataset temp cache to node-local fast storage (/tmp) to avoid Lustre locking
cache_dir = os.path.join(tempfile.gettempdir(), f"hf_cache_{os.getuid()}")
os.makedirs(cache_dir, exist_ok=True)

# --------------------------------------------------------------------
# Load Dataset (Stream directly without writing intermediate generator to disk)

print(f"Loading Dataset (using {num_cpus} CPU workers, cache dir: {cache_dir})")

raw_dataset = load_dataset(
    dataset_name,
    split=split,
    streaming=True
)

iterable_ds = raw_dataset.take(nrows)

# Convert directly to in-memory PyArrow dataset from generator stream
def gen():
    for item in iterable_ds:
        yield item

dataset = Dataset.from_generator(gen, cache_dir=cache_dir)

if True:
    dataset = dataset.shuffle(seed=1996)

# --------------------------------------------------------------------
# Tokenize

print("Tokenizing dataset...")

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', use_fast=True)
eos_token = tokenizer.eos_token

def tokenize_function(examples):
    # Process texts in batch using fast Rust tokenizer
    texts = [text + eos_token for text in examples["text"]]
    return tokenizer(
        texts,
        return_special_tokens_mask=False,
        return_attention_mask=False,
        add_special_tokens=False
    )

tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True,
    batch_size=2048,           # Larger batch size for fast Rust tokenization
    num_proc=num_cpus,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

# Save tokenizer
if split == 'train':
    os.makedirs(os.path.join(out_path, "tokenizer"), exist_ok=True)
    tokenizer.save_pretrained(os.path.join(out_path, "tokenizer"))

# --------------------------------------------------------------------
# Concat in chunks of max_seq_len

print("Concatenating in chunks of max_seq_len...")

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)] 
        for k, t in concatenated_examples.items()
    }
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,          # Reduced batch size to prevent Python list chain thrashing
    num_proc=num_cpus,
    desc="Grouping into chunks"
)

n_tokens = len(lm_datasets) * max_seq_length 
print(f"Number of tokens in dataset: {n_tokens:_}")

# --------------------------------------------------------------------
# Save Final Dataset to Workspace

if ordering == "randomized":
    lm_datasets = lm_datasets.shuffle(seed=96)

lm_datasets.set_format("torch")

final_save_path = os.path.join(out_path, split)
print(f"Saving final dataset to {final_save_path}...")
lm_datasets.save_to_disk(final_save_path)

print("Finished successfully!")
