#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:45:41 2024

@author: paveenhuang
"""

import os

os.environ['TRANSFORMERS_CACHE'] = '/data1/cache/d12922004/huggingface'
os.environ['HF_HOME'] = '/data1/cache/d12922004'

import torch
from transformers import AutoTokenizer, OPTForCausalLM
import pandas as pd
from typing import List
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import sys
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s", 
                    filename="embedding_extraction.log")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Get current GPU usage
def get_free_gpu():
    if not cuda_available:
        return None  # If no GPU, return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,index", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE, text=True
        )
        memory_usage = [int(line.split(",")[0]) for line in result.stdout.strip().split("\n")]
        return memory_usage.index(min(memory_usage))
    except Exception as e:
        logging.error(f"Error detecting GPUs: {e}")
        return None  

# Select device based on availability
if cuda_available:
    device_id = get_free_gpu()
    if device_id is not None:
        device = torch.device(f"cuda:{device_id}")
        logging.info(f"Using GPU: cuda:{device_id}")
    else:
        device = torch.device("cuda:0")
        logging.info("Falling back to GPU: cuda:0")
else:
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # If available, choose Apple's MPS
        logging.info("Using MPS device")
    else:
        device = torch.device("cpu")  # If no GPU and MPS, use CPU
        logging.info("Using CPU")

print(f"Using device: {device}")


def init_model(model_name: str):
    """
    Initialize the model and tokenizer, and move the model to the selected device.
    """
    try:
        model = OPTForCausalLM.from_pretrained(f"facebook/opt-{model_name}").to(device)
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-{model_name}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Model initialization error: {e}")
        sys.exit(1)

def load_data(dataset_path: Path, dataset_name: str, true_false: bool = False):
    """
    Load the dataset and handle exceptions.
    """
    filename_suffix = "_true_false" if true_false else ""
    dataset_file = dataset_path / f"{dataset_name}{filename_suffix}.csv"
    try:
        df = pd.read_csv(dataset_file)
        if "embeddings" not in df.columns:
            df["embeddings"] = pd.Series(dtype="object")
        return df
    except FileNotFoundError as e:
        logging.error(f"Dataset file not found: {e}")
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV: {e}")
    except pd.errors.EmptyDataError as e:
        logging.error(f"No data in CSV file: {e}")

def process_batch(batch_prompts: List[str], model, tokenizer, layers_to_use: list, remove_period: bool):
    """
    Batch process the data and return the embedded results.
    """
    if remove_period:
        batch_prompts = [prompt.rstrip(". ") for prompt in batch_prompts]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    seq_lengths = (inputs.attention_mask != 0).sum(dim=1) - 1
    batch_embeddings = {}

    for layer in layers_to_use:
        hidden_states = outputs.hidden_states[layer]
        last_hidden_states = hidden_states[range(hidden_states.size(0)), seq_lengths, :]
        batch_embeddings[layer] = [embedding.cpu().numpy().tolist() for embedding in last_hidden_states]

    return batch_embeddings


def save_data(df, output_path: Path, dataset_name: str, model_name: str, layer: int, remove_period: bool):
    """
    Save the processed data to a CSV file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_rmv_period" if remove_period else ""
    output_file = output_path / f"embeddings_{dataset_name}{model_name}_{abs(layer)}{filename_suffix}.csv"
    try:
        df.to_csv(output_file, index=False)
    except PermissionError:
        logging.error(f"Permission denied: {output_file}")
    except Exception as e:
        logging.error(f"Error saving file: {e}")

def main():
    """
    Load the configuration file, initialize the model, and process the dataset.
    """
    try:
        with open("config.json") as config_file:
            config_parameters = json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        return
    except PermissionError:
        logging.error("Permission denied.")
        return
    except json.JSONDecodeError:
        logging.error("Invalid JSON in config file.")
        return

    parser = argparse.ArgumentParser(description="Generate new CSV with embeddings.")
    parser.add_argument("--model", help="Name of the language model to use.")
    parser.add_argument("--layers", nargs="*", help="List of layers to use for embeddings.")
    parser.add_argument("--dataset_names", nargs="*", help="List of dataset names without CSV extension.")
    parser.add_argument("--true_false", type=bool, help="Append 'true_false' to dataset name?")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing.")
    parser.add_argument("--remove_period", type=bool, help="Remove periods at the end of sentences?")
    args = parser.parse_args()

    model_name = args.model or config_parameters["model"]
    should_remove_period = args.remove_period if args.remove_period is not None else config_parameters["remove_period"]
    layers_to_process = [int(x) for x in (args.layers or config_parameters["layers_to_use"])]
    dataset_names = args.dataset_names or config_parameters["list_of_datasets"]
    true_false = args.true_false if args.true_false is not None else config_parameters["true_false"]
    batch_size = args.batch_size or config_parameters["batch_size"]
    dataset_path = Path(config_parameters["dataset_path"])
    output_path = Path(config_parameters["processed_dataset_path"])

    model, tokenizer = init_model(model_name)

    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        dataset = load_data(dataset_path, dataset_name, true_false=true_false)
        if dataset is None:
            continue

        num_batches = len(dataset) // batch_size + (len(dataset) % batch_size != 0)

        model_output_per_layer = {layer: dataset.copy() for layer in layers_to_process}
        for layer in layers_to_process:
            model_output_per_layer[layer]["embeddings"] = pd.Series(dtype="object")

        for batch_num in tqdm(range(num_batches), desc=f"Processing batches in {dataset_name}"):
            try:
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                batch = dataset.iloc[start_idx:end_idx]
                batch_prompts = batch["statement"].tolist()
                batch_embeddings = process_batch(batch_prompts, model, tokenizer, layers_to_process, should_remove_period)

                for layer in layers_to_process:
                    for i, idx in enumerate(range(start_idx, end_idx)):
                        model_output_per_layer[layer].at[idx, "embeddings"] = batch_embeddings[layer][i]

                if batch_num % 10 == 0:
                    logging.info(f"Processed batch {batch_num} of {num_batches}")

            except Exception as e:
                logging.error(f"Error in batch {batch_num}: {e}")

        for layer in layers_to_process:
            save_data(model_output_per_layer[layer], output_path, dataset_name, model_name, layer, should_remove_period)

if __name__ == "__main__":
    main()