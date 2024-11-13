#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:31:39 2024

@author: paveenhuang
"""


def get_token_embeddings(statement, model, tokenizer, layer, max_seq_length=None, statement_index=0):
    """Process a single statement and get per-token embeddings."""
    results = []
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

    hidden_states = outputs.hidden_states[layer]
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
        if prefix and token.startswith(prefix) and current_word_tokens:
            # When encountering a new word, aggregate the token embeddings of the previous word
            word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
            word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
            results.append({"word": word_text, "embedding": word_embedding.cpu().numpy()})
            current_word_tokens = [token.lstrip("Ġ")]
            current_word_embeddings = [hidden_state[j]]
        else:
            # Continue to add to the current word
            current_word_tokens.append(token.lstrip(prefix))
            current_word_embeddings.append(hidden_state[j])

    # Process the last word
    if current_word_tokens:
        word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
        word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
        results.append({"word": word_text, "embedding": word_embedding.cpu().numpy()})

    return results


def probe(statements, model_name, layer, token=None):
    """
    Main function to extract words and their predictions.
    """
    config_parameters = load_config()
    probes_path = Path(config_parameters["probes_dir"])
    token =config_parameters.get("token") # huggingface token
    language_model, tokenizer = init_model(model_name, token)
    max_seq_length = language_model.config.max_position_embeddings
    
    sanitized_model_name = model_name.replace("/", "_")

    all_words_results = []  # Store lists of words for each statement
    

    for idx, statement in enumerate(statements):
        embeddings_data = get_token_embeddings(
            statement, language_model, tokenizer, layer, max_seq_length=max_seq_length, statement_index=idx
        )

        embeddings = np.array([item["embedding"] for item in embeddings_data])
        words = [item["word"] for item in embeddings_data]

        all_words_results.append(words)  # Append word list for each statement

        # Load the trained model path
        probe_model_path = get_probe_path(probes_path, sanitized_model_name, layer)

        # Initialize the probe model with the correct input_dim
        input_dim = embeddings.shape[1]
        probe_model = SAPLMAClassifier(input_dim=input_dim).to(language_model.device)

        # Load the trained weights
        probe_model.load_state_dict(torch.load(probe_model_path, map_location=language_model.device))
        probe_model.eval()

        # Load the optimal threshold
        threshold_file = "threshold.json"
        probe_name = f"{sanitized_model_name}_{abs(layer)}_combined"
        threshold = load_threshold(threshold_file, probe_name)

        # Convert embeddings to tensor
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(language_model.device)

        # Make predictions
        with torch.no_grad():
            outputs = probe_model(embeddings_tensor)
            probabilities = outputs.cpu().numpy().ravel()
            predictions = (probabilities >= threshold).astype(int)

        # Add predictions to words
        for i, word in enumerate(words):
            all_words_results[idx][i] = {
                "word": word,
                "predicted_probability": probabilities[i],
                "predicted_label": predictions[i],
            }

    return all_words_results  # Return a list of word lists with predictions


if __name__ == "__main__":
    import os
    import torch
    import numpy as np
    from pathlib import Path
    import logging
    from model import SAPLMAClassifier
    from utils import init_model, load_config, get_probe_path, load_threshold

    os.environ["HF_HOME"] = "/data1/cache/d12922004"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="embedding_extraction.log")

    model_name = "facebook/opt-350m"
    layer = -4
    statements = [
        "The capital of Australia is Sydney.",
        "Birds lay eggs.",
    ]

    result = probe(statements, model_name, layer)
    print(result)
