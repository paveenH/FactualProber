#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:06:53 2024

@author: paveenhuang
"""

import torch
from transformers import AutoTokenizer, OPTForCausalLM
import numpy as np
import json
import argparse
from pathlib import Path
from statements import statements, labels
import logging
from model import SAPLMAClassifier  
from tqdm import tqdm
import pandas as pd

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="prediction.log",
)

# Check GPU availability
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logging.info(f"Using device: {device}")


def load_config(config_file):
    """Load configuration from JSON file."""
    try:
        with open(config_file) as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        logging.error(f"Config file {config_file} not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error parsing JSON in {config_file}.")
        raise


def init_model(model_name: str):
    """
    Initialize the language model and tokenizer.
    """
    try:
        model_name_full = f"facebook/opt-{model_name}"
        model = OPTForCausalLM.from_pretrained(
            model_name_full, output_hidden_states=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_full)
        logging.info(f"Model {model_name_full} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Model initialization error: {e}")
        raise


def get_embeddings(
    statements,
    model,
    tokenizer,
    layers_to_use,
    batch_size=8,
    max_seq_length=None,
):
    """
    Process statements in batches and get embeddings.
    """
    embeddings = []

    num_batches = (len(statements) + batch_size - 1) // batch_size

    for batch_num in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(statements))
        batch_statements = statements[start_idx:end_idx]

        inputs = tokenizer(
            batch_statements,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        seq_lengths = (inputs.attention_mask != 0).sum(dim=1) - 1

        for layer in layers_to_use:
            hidden_states = outputs.hidden_states[layer]
            batch_embeddings = hidden_states[
                torch.arange(hidden_states.size(0)), seq_lengths, :
            ]
            embeddings.append(batch_embeddings.cpu().numpy())

    # Concatenate embeddings from all batches
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def load_trained_model_path(probes_path, model_name, layer):
    """
    Get the path of the trained probe model.
    """
    model_path = probes_path / f"{model_name}_{abs(layer)}_combined.pt"
    if not model_path.exists():
        logging.error(f"Trained model not found at {model_path}")
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    logging.info(f"Trained model found at {model_path}")
    return model_path


def load_threshold(threshold_file, probe_name):
    """
    Load the optimal threshold from a JSON file.
    """
    try:
        with open(threshold_file, 'r') as f:
            thresholds = json.load(f)
        threshold = thresholds.get(probe_name, 0.5)
        logging.info(f"Optimal threshold for {probe_name} loaded: {threshold}")
        return threshold
    except FileNotFoundError:
        logging.error(f"Threshold file {threshold_file} not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error parsing JSON in {threshold_file}.")
        raise


def main():
    # Load config
    config_parameters = load_config("config.json")

    # Argument parsing
    parser = argparse.ArgumentParser(description="Make predictions on new statements.")
    parser.add_argument("--model", help="Model name.")
    parser.add_argument("--layer", type=int, help="Layer for embeddings.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing.")
    args = parser.parse_args()

    model_name = args.model if args.model is not None else config_parameters["model"]
    layer = args.layer if args.layer is not None else config_parameters["layers_to_use"][0]
    batch_size = args.batch_size

    probes_path = Path(config_parameters["probes_dir"])

    # Initialize model and tokenizer
    language_model, tokenizer = init_model(model_name)
    max_seq_length = language_model.config.max_position_embeddings

    # Get embeddings
    embeddings = get_embeddings(
        statements,
        language_model,
        tokenizer,
        layers_to_use=[layer],
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    # Get input_dim from embeddings
    input_dim = embeddings.shape[1]

    # Load the trained model path
    model_path = load_trained_model_path(probes_path, model_name, layer)

    # Initialize the probe model with the correct input_dim
    probe_model = SAPLMAClassifier(input_dim=input_dim).to(device)

    # Load the trained weights
    probe_model.load_state_dict(torch.load(model_path, map_location=device))
    probe_model.eval()

    # Load the optimal threshold
    threshold_file = 'threshold.json'
    probe_name = f"{model_name}_{abs(layer)}_combined"
    threshold = load_threshold(threshold_file, probe_name)

    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = probe_model(embeddings_tensor)
        probabilities = outputs.cpu().numpy().ravel()
        predictions = (probabilities >= threshold).astype(int)

    # Optionally, save the results to a file
    results = []
    for idx in range(len(statements)):
        results.append({
            'statement': statements[idx],
            'true_label': labels[idx],
            'predicted_probability': probabilities[idx],
            'predicted_label': predictions[idx]
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('prediction_results.csv', index=False)


if __name__ == "__main__":
    main()