#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:58:07 2024

@author: paveenhuang
"""

import sys
import json
import logging
import torch
import pandas as pd
import subprocess

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_config(json_path="config.json"):
    """
    Load the configuration file, initialize the model, and process the dataset.
    """
    try:
        with open(json_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        return
    except PermissionError:
        logging.error("Permission denied.")
        return
    except json.JSONDecodeError:
        logging.error("Invalid JSON in config file.")
        return


def load_threshold(threshold_file, probe_name):
    """
    Load the optimal threshold from a JSON file.
    """
    try:
        with open(threshold_file, "r") as f:
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
        
        
def load_data(dataset_path: Path, dataset_name: str, true_false: bool = False):
    """
    Load the dataset and handle exceptions.
    """
    filename_suffix = "_true_false" if true_false else ""
    dataset_file = dataset_path / f"{dataset_name}{filename_suffix}.csv"
    df = pd.read_csv(dataset_file)
    return df



def get_probe_path(probe_path, model_name, layer):
    """
    Get the path of the trained probe model.
    """
    probe_path = probe_path / f"{model_name}_{abs(layer)}_combined.pt"
    if not probe_path.exists():
        logging.error(f"Trained model not found at {probe_path}")
        raise FileNotFoundError(f"Trained model not found at {probe_path}")
    logging.info(f"Trained model found at {probe_path}")
    return probe_path


def select_device():
    """
    Select the appropriate device: GPU (CUDA), Apple MPS, or CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using GPU: cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def get_free_gpu(cuda_available: bool):
    if not cuda_available:
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,index", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            text=True
        )
        memory_usage = [int(line.split(",")[0]) for line in result.stdout.strip().split("\n")]
        return memory_usage.index(min(memory_usage))
    except Exception as e:
        logging.error(f"Error detecting GPUs: {e}")
        return None


def init_model(model_name: str, token: str, use_fp: int = 16):
    """
    Initialize the model and tokenizer with automatic device mapping and specified precision.
    Supports both OPT and Llama models.

    Parameters:
    - model_name (str): The name of the model to load.
    - token (str): Hugging Face authentication token.
    - use_fp (int): Precision mode. Accepts 8, 16, or 32.
                     8 for 8-bit quantization,
                     16 for FP16,
                     32 for FP32.
    """
    logging.info(f"Loading model: {model_name}")

    is_llama = "llama" in model_name.lower()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=token, 
        trust_remote_code=is_llama)
    logging.info("Tokenizer loaded successfully.")

    # Set pad_token to eos_token (if the model requires it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_fp == 16:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float16, 
            token=token, 
            trust_remote_code=is_llama
        )
        logging.info("Model loaded successfully with FP16 precision.")
    # elif use_fp == 8:
    #     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         device_map="auto",
    #         quantization_config=quantization_config,
    #         token=token, 
    #         trust_remote_code=is_llama
    #         )
    #     logging.info("Model loaded successfully with 8-bit quantization.")
    elif use_fp == 32:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float32, 
            use_auth_token=token, 
            trust_remote_code=is_llama
        )
        logging.info("Model loaded successfully with FP32 precision.")
    else:
        logging.error("Invalid value for use_fp. Choose from 8, 16, or 32.")
        sys.exit(1)
    return model, tokenizer
