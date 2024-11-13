#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:58:51 2024

@author: paveenhuang
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
from statements import statements, labels
import logging
from model import SAPLMAClassifier
import pandas as pd
from utils import init_model, load_config, load_threshold, select_device


def get_token_embeddings(statement, model, tokenizer, layer, max_seq_length=None, statement_index=0):
    """
    Process a single statement and get per-token embeddings.
    """
    results = []

    # Convert statement to input format
    inputs = tokenizer(
        statement,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
        add_special_tokens=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Select the hidden state of the specified layer (batch_size=1, seq_length, hidden_size)
    hidden_states = outputs.hidden_states[layer]

    # Extract token and its embedding
    input_ids = inputs.input_ids[0]
    attention_mask = inputs.attention_mask[0]
    seq_len = attention_mask.sum().item()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[:seq_len])
    hidden_state = hidden_states[0, :seq_len, :]

    # Check tokenizer type for handling token prefixes
    if hasattr(tokenizer, "word_tokenizer"):
        # For Llama and similar tokenizers
        prefix = "▁"
    else:
        # For GPT-2 and similar tokenizers
        prefix = "Ġ"
        
    # Traverse each token, aggregate tokens by word and calculate their embeddings
    current_word_tokens = []
    current_word_embeddings = []

    for j, token in enumerate(tokens):
        if token.startswith(prefix) and current_word_tokens:
            # When encountering a new word, aggregate the token embeddings of the previous word
            word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
            word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
            results.append({"statement_index": statement_index, "word": word_text, "embedding": word_embedding.cpu().numpy()})

            # Initialize the new word
            current_word_tokens = [token.lstrip(prefix)]
            current_word_embeddings = [hidden_state[j]]
        else:
            # Continue to add to the current word
            current_word_tokens.append(token.lstrip(prefix))
            current_word_embeddings.append(hidden_state[j])

    # Process the last word
    if current_word_tokens:
        word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
        word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
        results.append({"statement_index": statement_index, "word": word_text, "embedding": word_embedding.cpu().numpy()})

    return results


def main():
    # Load config
    config_parameters = load_config()

    # Argument parsing
    parser = argparse.ArgumentParser(description="Make predictions on new statements.")
    parser.add_argument("--model", help="Model name.")
    parser.add_argument("--layers", type=int, help="Layer for embeddings.")
    parser.add_argument("--token", help="Your Hugging Face access token.")
    parser.add_argument("--probe_name", help="Probe name")
    args = parser.parse_args()

    model_name = args.model if args.model is not None else config_parameters["model"]
    probe_name = args.probe_name or config_parameters.get("probe_name")
    token = args.token if args.token is not None else config_parameters.get("token")
    layers_to_process = [int(x) for x in (args.layers or config_parameters["layers_to_use"])]

    layer = layers_to_process[0]
    sanitized_model_name = model_name.replace("/", "_")
    probe_name = f"{sanitized_model_name}_{abs(layer)}_{probe_name}"

    probe_path = Path(config_parameters["probes_dir"])

    # Initialize model and tokenizer
    language_model, tokenizer = init_model(model_name, token)
    max_seq_length = language_model.config.max_position_embeddings

    all_embeddings = []
    all_words = []
    all_statement_indices = []

    for idx, statement in enumerate(statements):
        embeddings_data = get_token_embeddings(
            statement, language_model, tokenizer, layer, max_seq_length=max_seq_length, statement_index=idx
        )

        embeddings = np.array([item["embedding"] for item in embeddings_data])
        words = [item["word"] for item in embeddings_data]
        statement_indices = [item["statement_index"] for item in embeddings_data]

        all_embeddings.append(embeddings)
        all_words.extend(words)
        all_statement_indices.extend(statement_indices)

    # Concatenate embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)

    # Get input_dim from embeddings
    input_dim = embeddings.shape[1]

    # Load the trained model
    probe_path = probe_path / f"{probe_name}.pt"
    print(f"Trained model found at {probe_path}")
    probe_model = SAPLMAClassifier(input_dim=input_dim).to(language_model.device)
    probe_model.load_state_dict(torch.load(probe_path, map_location=language_model.device))
    probe_model.eval()
    print("probe name", probe_path)

    # Load the optimal threshold
    threshold_file = "threshold.json"
    threshold = load_threshold(threshold_file, probe_name)
    print("Load threshold successfully, threshold ", threshold)

    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(language_model.device)

    # Make predictions
    with torch.no_grad():
        outputs = probe_model(embeddings_tensor)
        probabilities = outputs.cpu().numpy().ravel()
        predictions = (probabilities >= threshold).astype(int)

    # Optionally, save the results to a file
    results = []
    for idx in range(len(all_words)):
        statement_idx = all_statement_indices[idx]
        results.append(
            {
                "statement_index": statement_idx,
                "statement": statements[statement_idx],
                "word": all_words[idx],
                "true_label": labels[statement_idx],
                "predicted_probability": probabilities[idx],
                "predicted_label": predictions[idx],
            }
        )
    results_df = pd.DataFrame(results)
    results_dir = os.path.join(os.getcwd(), "prediction")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"prediction_results_{probe_name}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":

    # Set HF_HOME
    os.environ["HF_HOME"] = "/data1/cache/d12922004"
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="embedding_extraction.log")

    # Select device
    device = select_device()
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    main()
