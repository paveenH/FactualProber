#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:45:09 2024

@author: paveenhuang
"""

import subprocess
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import logging
import sys
import argparse
from sklearn.model_selection import train_test_split
from copy import deepcopy

from model import AttentionMLP
from utils import load_config


def get_free_gpu():
    if not cuda_available:
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,index", "--format=csv,nounits,noheader"], stdout=subprocess.PIPE, text=True
        )
        memory_usage = [int(line.split(",")[0]) for line in result.stdout.strip().split("\n")]
        return memory_usage.index(min(memory_usage))
    except Exception as e:
        logging.error(f"Error detecting GPUs: {e}")
        return None


def load_datasets(dataset_names, remove_period, input_path, sanitized_model_name):
    datasets = []
    for dataset_name in dataset_names:
        path_suffix = "_rmv_period" if remove_period else ""
        path = input_path / f"embeddings_{dataset_name}_{sanitized_model_name}{path_suffix}.csv"
        datasets.append(pd.read_csv(path))
    return datasets
            

def correct_str(str_arr):
    """Converts a string representation of a numpy array into a comma-separated string."""
    val_to_ret = (
        str_arr.replace("[array(", "")
        .replace("dtype=float32)]", "")
        .replace("\n", "")
        .replace(" ", "")
        .replace("],", "]")
        .replace("[", "")
        .replace("]", "")
    )
    return val_to_ret


def compute_roc_auc(y_true, y_scores):
    """Compute ROC AUC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr, thresholds


def find_optimal_threshold(fpr, tpr, thresholds):
    """Find the optimal threshold that maximizes the difference between TPR and FPR."""
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def extract_embeddings_and_labels(dataset):
    """Extract embeddings and labels with consistency checks."""
    embeddings = []
    labels = []
    
    for i, embedding_str in enumerate(dataset["embeddings"].tolist()):
        embedding = json.loads(embedding_str)
        labels.append(dataset["label"].iloc[i])
        embeddings.append(embedding)    
    # embeddings = np.array(embeddings, dtype=object)
    embeddings = np.stack([np.array(sample) for sample in embeddings])
    labels = np.array(labels)
    return embeddings, labels


def save_threshold(threshold_file, probe_name, threshold_value):
    """Save the optimal threshold to a JSON file."""
    try:
        if os.path.exists(threshold_file):
            with open(threshold_file, "r") as f:
                thresholds = json.load(f)
        else:
            thresholds = {}
        thresholds[probe_name] = threshold_value
        with open(threshold_file, "w") as f:
            json.dump(thresholds, f, indent=4)
        print(f"Optimal threshold saved to {threshold_file}")
    except Exception as e:
        print(f"Error saving threshold: {e}")
        sys.exit(1)


def train_layers(model, train_embeddings, train_labels, val_embeddings, val_labels, epochs=5, batch_size=32, learning_rate=0.001):
    """Train the model and evaluate on validation set after each epoch."""
    # Prepare training data
    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare validation data
    val_embeddings_tensor = torch.tensor(val_embeddings, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1)

    val_dataset = TensorDataset(val_embeddings_tensor, val_labels_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()  # Set model to training mode
        for batch_embeddings, batch_labels in train_dataloader:
            batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}", end="")

        # Evaluate on validation set
        model.eval()  # Set model to evaluation mode
        all_val_outputs = []
        all_val_labels = []
        with torch.no_grad():
            for val_embeddings_batch, val_labels_batch in val_dataloader:
                val_embeddings_batch = val_embeddings_batch.to(device)
                val_labels_batch = val_labels_batch.to(device)
                val_outputs = model(val_embeddings_batch)
                all_val_outputs.append(val_outputs.cpu())
                all_val_labels.append(val_labels_batch.cpu())

        val_outputs_cat = torch.cat(all_val_outputs).numpy()
        val_labels_cat = torch.cat(all_val_labels).numpy()

        val_preds = (val_outputs_cat >= 0.5).astype(int)
        val_accuracy = accuracy_score(val_labels_cat, val_preds)
        print(f", Validation Accuracy: {val_accuracy:.4f}")

        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = deepcopy(model.state_dict())
            logging.info(f"New best validation accuracy: {best_val_accuracy:.4f} at epoch {epoch + 1}")

    # Load the best model weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, test_embeddings, test_labels, batch_size=32):
    """Evaluate the model and return predictions and probabilities."""
    test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    model.eval()
    model.to(device)
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            all_probs.append(outputs.cpu())
            all_labels.append(batch_labels.cpu())

    avg_loss = total_loss / len(dataloader)
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return avg_loss, all_probs, all_labels


def main():
    # Set up logging
    try:
        logging.basicConfig(
            filename="classification.log", level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(__name__)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return
    logger.info("Execution started.")

    # Load config
    config_parameters = load_config()

    # Argument parsing
    parser = argparse.ArgumentParser(description="Train probes on processed and labeled datasets.")
    parser.add_argument("--model", help="Model name.")
    parser.add_argument("--dataset_names", nargs="*", help="Dataset names.")
    parser.add_argument("--remove_period", action="store_true", help="Remove final period from sentences.")
    parser.add_argument("--save_probes", action="store_true", help="Save trained probes.")
    parser.add_argument("--probe_name", help="Probe name")
    
    args = parser.parse_args()

    model_name = args.model or config_parameters.get("model")
    probe_name = args.probe_name or config_parameters.get("probe_name")
    remove_period = args.remove_period if args.remove_period else config_parameters.get("remove_period", False)
    dataset_names = args.dataset_names if args.dataset_names is not None else config_parameters.get("list_of_datasets", [])
    save_probes = args.save_probes if args.save_probes else config_parameters.get("save_probes", False)
    input_path = Path(config_parameters.get("processed_dataset_path", "embeddings"))
    probes_path = Path(config_parameters.get("probes_dir", "probes"))

    sanitized_model_name = model_name.replace("/", "_")

    # Load datasets
    datasets = load_datasets(dataset_names, remove_period, input_path, sanitized_model_name)
    combined_dataset = pd.concat(datasets, ignore_index=True)

    # Split combined_dataset into train, val, test
    train_dataset, temp_dataset = train_test_split(combined_dataset, test_size=0.2, random_state=42, stratify=combined_dataset["label"])
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42, stratify=temp_dataset["label"])

    # Count the number of samples in each split
    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)

    print(f"Number of samples in training set: {num_train_samples}")
    print(f"Number of samples in validation set: {num_val_samples}")
    print(f"Number of samples in test set: {num_test_samples}")

    # Display label distribution in each split
    train_label_counts = train_dataset["label"].value_counts().to_dict()
    val_label_counts = val_dataset["label"].value_counts().to_dict()
    test_label_counts = test_dataset["label"].value_counts().to_dict()

    print(f"Label distribution in training set: {train_label_counts}")
    print(f"Label distribution in validation set: {val_label_counts}")
    print(f"Label distribution in test set: {test_label_counts}")

    # Extract embeddings and labels
    train_embeddings, train_labels = extract_embeddings_and_labels(train_dataset)
    val_embeddings, val_labels = extract_embeddings_and_labels(val_dataset)
    test_embeddings, test_labels = extract_embeddings_and_labels(test_dataset)

    # Train the model
    num_layers = train_embeddings.shape[1]
    hidden_size = train_embeddings.shape[2]

    model = AttentionMLP(hidden_size=hidden_size, num_layers=num_layers, num_heads=8, dropout=0.1).to(device)
    model = train_layers(model, train_embeddings, train_labels, val_embeddings, val_labels)

    # Evaluate on validation set to find optimal threshold
    val_loss, val_probs, val_true = evaluate_model(model, val_embeddings, val_labels)
    val_probs_flat = val_probs.ravel()
    val_true_flat = val_true.ravel()

    # Compute ROC AUC on validation set
    val_roc_auc, val_fpr, val_tpr, val_thresholds = compute_roc_auc(val_true_flat, val_probs_flat)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(val_fpr, val_tpr, val_thresholds)

    # Convert optimal_threshold to native Python float
    optimal_threshold_value = float(optimal_threshold)

    # Save the optimal threshold to threshold.json
    probe_name_full = f"{sanitized_model_name}_{probe_name}"
    threshold_file = "threshold.json"
    save_threshold(threshold_file, probe_name_full, optimal_threshold_value)

    # Evaluate on test set
    test_loss, test_probs, test_true = evaluate_model(model, test_embeddings, test_labels)
    test_probs_flat = test_probs.ravel()
    test_true_flat = test_true.ravel()

    # Calculate accuracy with optimal threshold
    test_pred_labels = (test_probs_flat >= optimal_threshold_value).astype(int)
    test_accuracy = accuracy_score(test_true_flat, test_pred_labels)
    test_roc_auc, _, _, _ = compute_roc_auc(test_true_flat, test_probs_flat)

    # Print results
    print(f"Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_roc_auc:.4f}, Optimal Threshold: {optimal_threshold_value:.4f}")

    # Save the trained model
    if save_probes:
        os.makedirs(probes_path, exist_ok=True)
        probe_path = probes_path / f"{probe_name_full}.pt"
        torch.save(model.state_dict(), probe_path)
        logging.info(f"Trained model saved to {probe_path}")

    # Save test predictions and probabilities
    test_dataset_copy = test_dataset.copy()

    # Remove the 'embeddings' column
    if "embeddings" in test_dataset_copy.columns:
        test_dataset_copy = test_dataset_copy.drop(columns=["embeddings"])

    test_dataset_copy["probability"] = test_probs_flat
    test_dataset_copy["prediction"] = test_pred_labels
    test_dataset_copy["probability"] = test_dataset_copy["probability"].round(4)

    # Save to CSV
    output_dir = os.path.join(os.getcwd(), "prediction")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{probe_name_full}_predictions.csv")
    test_dataset_copy.to_csv(output_path, index=False)

    logger.info("Execution completed.")


if __name__ == "__main__":
    os.environ["HF_HOME"] = "/data1/cache/d12922004"

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="embedding_extraction.log")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()

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

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    main()
