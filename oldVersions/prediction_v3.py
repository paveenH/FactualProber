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
import pandas as pd
import subprocess
import os

os.environ['HF_HOME'] = '/data1/cache/d12922004'

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
    

def get_token_embeddings(
    statement,
    model,
    tokenizer,
    layer,
    max_seq_length=None,
    statement_index=0
):
    """
    Process a single statement and get per-token embeddings.
    """
    results = []

    # 将 statement 转换为输入格式
    inputs = tokenizer(
        statement,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
        add_special_tokens=False,
    ).to(device)

    # 获取模型的输出
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # 选取指定层的隐藏状态 (batch_size=1, seq_length, hidden_size)
    hidden_states = outputs.hidden_states[layer]

    # 提取 token 和其嵌入
    input_ids = inputs.input_ids[0]  # 获取第一个（也是唯一一个）句子的 token ids
    attention_mask = inputs.attention_mask[0]  # 获取对应的 attention mask
    seq_len = attention_mask.sum().item()  # 有效 token 的长度

    tokens = tokenizer.convert_ids_to_tokens(input_ids[:seq_len])
    hidden_state = hidden_states[0, :seq_len, :]  # 获取该句子的所有 token 的嵌入

    # 遍历每个 token，按词聚合 token 并计算其嵌入
    current_word_tokens = []
    current_word_embeddings = []

    for j, token in enumerate(tokens):
        if token.startswith('Ġ') and current_word_tokens:
            # 遇到新词，聚合前一个词的 token 嵌入
            word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
            word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
            results.append({
                'statement_index': statement_index,
                'word': word_text,
                'embedding': word_embedding.cpu().numpy()
            })

            # 初始化新词
            current_word_tokens = [token.lstrip('Ġ')]
            current_word_embeddings = [hidden_state[j]]
        else:
            # 继续添加到当前词
            current_word_tokens.append(token.lstrip('Ġ'))
            current_word_embeddings.append(hidden_state[j])

    # 处理最后一个词
    if current_word_tokens:
        word_embedding = torch.stack(current_word_embeddings).mean(dim=0)
        word_text = tokenizer.convert_tokens_to_string(current_word_tokens)
        results.append({
            'statement_index': statement_index,
            'word': word_text,
            'embedding': word_embedding.cpu().numpy()
        })
        
    return results


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
    args = parser.parse_args()

    model_name = args.model if args.model is not None else config_parameters["model"]
    layer = args.layer if args.layer is not None else config_parameters["layer"]

    probes_path = Path(config_parameters["probes_dir"])

    # Initialize model and tokenizer
    language_model, tokenizer = init_model(model_name)
    max_seq_length = language_model.config.max_position_embeddings
    
    all_embeddings = []
    all_words = []
    all_statement_indices = []

    for idx, statement in enumerate(statements):
        embeddings_data = get_token_embeddings(
            statement,
            language_model,
            tokenizer,
            layer,
            max_seq_length=max_seq_length,
            statement_index=idx
            )

        embeddings = np.array([item['embedding'] for item in embeddings_data])
        words = [item['word'] for item in embeddings_data]
        statement_indices = [item['statement_index'] for item in embeddings_data]

        all_embeddings.append(embeddings)
        all_words.extend(words)
        all_statement_indices.extend(statement_indices)

    # Concatenate embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)

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
    for idx in range(len(all_words)):
        statement_idx = all_statement_indices[idx]
        results.append({
            'statement_index': statement_idx,
            'statement': statements[statement_idx],
            'word': all_words[idx],
            'true_label': labels[statement_idx],
            'predicted_probability': probabilities[idx],
            'predicted_label': predictions[idx]
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'prediction_results_{model_name}.csv', index=False)

if __name__ == "__main__":
    main()