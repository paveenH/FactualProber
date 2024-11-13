#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 22:54:21 2024

@author: paveenhuang
"""

import os
import torch
import logging
from typing import List
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

from utils import init_model, load_config, load_data


def process_batch(batch_prompts: List[str], model, tokenizer, remove_period: bool):
    """
    Batch process the data and return the embeddings from all Transformer layers for the last token.
    """
    if remove_period:
        batch_prompts = [prompt.rstrip(". ") for prompt in batch_prompts]

    # Tokenize inputs
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Number of layers (excluding embedding layer)
    batch_size = inputs["input_ids"].size(0)
    seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # Shape: (batch_size)

    hidden_dims = []
    hidden_states = []

    # Collect hidden dimensions for all layers
    for layer_idx, layer in enumerate(outputs.hidden_states[1:], start=1):  # Start from 1 to skip embedding layer
        # layer shape: (batch_size, seq_length, hidden_dim)
        last_token_hidden_states = layer[torch.arange(batch_size, device=model.device), seq_lengths, :]
        hidden_dim = last_token_hidden_states.size(1)
        hidden_dims.append(hidden_dim)
    max_hidden_size = max(hidden_dims)

    # Second pass: Extract embeddings and pad/truncate to max_hidden_size
    for layer_idx, layer in enumerate(outputs.hidden_states[1:], start=1):
        # layer shape: (batch_size, seq_length, hidden_dim)
        last_token_hidden_states = layer[torch.arange(batch_size, device=model.device), seq_lengths, :]
        last_token_hidden_states_np = last_token_hidden_states.cpu().numpy()
        hidden_dim = last_token_hidden_states_np.shape[1]

        if hidden_dim < max_hidden_size:
            padding_width = max_hidden_size - hidden_dim
            padding = np.zeros((batch_size, padding_width), dtype=last_token_hidden_states_np.dtype)
            last_token_hidden_states_padded = np.hstack((last_token_hidden_states_np, padding))
            logging.debug(f"Layer {layer_idx}: Hidden dim {hidden_dim} padded to {max_hidden_size}.")
        else:
            last_token_hidden_states_padded = last_token_hidden_states_np

        hidden_states.append(last_token_hidden_states_padded.tolist())
        logging.debug(f"Layer {layer_idx} embeddings extracted.")

    # Transpose to (batch_size, num_layers, hidden_dim)
    batch_embeddings = list(zip(*hidden_states))
    batch_embeddings = [list(layer_embeddings) for layer_embeddings in batch_embeddings]

    # Clear the cache to free up GPU memory
    torch.cuda.empty_cache()

    return batch_embeddings


def save_data(df, embeddings, output_path: Path, dataset_name: str, model_name: str, remove_period: bool):
    """
    Save the processed data to a CSV file with embeddings as JSON strings.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    sanitized_model_name = model_name.replace("/", "_")
    output_file = output_path / f"embeddings_{dataset_name}_{sanitized_model_name}"
    
    embeddings_array = np.array(embeddings, dtype=np.float16)
    np.save(output_file, embeddings_array)
    

def main():
    config_parameters = load_config()

    parser = argparse.ArgumentParser(description="Generate new CSV with embeddings.")
    parser.add_argument(
        "--model",
        help="Full name of the language model to use (e.g., 'facebook/opt-2.7b', 'meta-llama/Llama-3.2-8B-Instruct').",
    )
    parser.add_argument("--dataset_names", nargs="*", help="List of dataset names without CSV extension.")
    parser.add_argument("--true_false", type=bool, help="Append 'true_false' to dataset name?")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing.")
    parser.add_argument("--remove_period", type=bool, help="Remove periods at the end of sentences?")
    parser.add_argument("--token", help="Your Hugging Face access token.")

    args = parser.parse_args()

    # Get parameters from args or config
    token = args.token or config_parameters.get("token")
    model_name = args.model or config_parameters.get("model")
    should_remove_period = args.remove_period if args.remove_period is not None else config_parameters.get("remove_period", False)
    dataset_names = args.dataset_names or config_parameters.get("list_of_datasets", [])
    true_false = args.true_false if args.true_false is not None else config_parameters.get("true_false", False)
    batch_size = args.batch_size or config_parameters.get("batch_size", 32)
    dataset_path = Path(config_parameters.get("dataset_path", "datasets"))
    output_path = Path(config_parameters.get("processed_dataset_path", "embeddings"))

    # Initialize model and tokenizer
    model, tokenizer = init_model(model_name, token)
    model.eval()  # Set model to evaluation mode

    logging.info("Execution started.")

    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        dataset = load_data(dataset_path, dataset_name, true_false=true_false)
        if dataset is None:
            continue

        num_samples = len(dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size

        embeddings = []

        for batch_num in tqdm(range(num_batches), desc=f"Processing batches in {dataset_name}"):
            try:
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch = dataset.iloc[start_idx:end_idx]
                batch_prompts = batch["statement"].tolist()
                batch_embeddings = process_batch(
                    batch_prompts,
                    model,
                    tokenizer,
                    remove_period=should_remove_period,
                )
                embeddings.extend(batch_embeddings)

                if (batch_num + 1) % 10 == 0:
                    logging.info(f"Processed batch {batch_num + 1} of {num_batches}")
            except Exception as e:
                logging.error(f"Error in batch {batch_num + 1}: {e}")

        # Save embeddings
        save_data(dataset, embeddings, output_path, dataset_name, model_name, should_remove_period)


if __name__ == "__main__":
    # set HF_HOME
    os.environ["HF_HOME"] = "/data1/cache/d12922004"

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="embedding_extraction.log",
    )

    main()
