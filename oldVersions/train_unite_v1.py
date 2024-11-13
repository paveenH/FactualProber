#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:16:07 2024

@author: paveenhuang
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import logging
import os
import sys
import argparse
from sklearn.model_selection import train_test_split
from copy import deepcopy


from model import SAPLMAClassifier  # Ensure this is correctly imported


# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU.")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
    print("Using MPS (Apple Silicon GPU).")
else:
    device = torch.device('cpu')
    print("Using CPU.")

# Set random seed for reproducibility
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)


def load_config(config_file):
    """Load configuration from JSON file."""
    try:
        with open(config_file) as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"Config file {config_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error parsing JSON in {config_file}.")
        sys.exit(1)


def load_datasets(dataset_names, layers_to_process, remove_period, input_path, model_name, idx):
    """Load embeddings from csv files."""
    datasets = []
    dataset_paths = []
    for dataset_name in dataset_names:
        try:
            path_suffix = "_rmv_period" if remove_period else ""
            path = input_path / f"embeddings_{dataset_name}{model_name}_{abs(layers_to_process[idx])}{path_suffix}.csv"
            datasets.append(pd.read_csv(path))
            dataset_paths.append(path)
        except FileNotFoundError:
            print(f"File not found: {path}. Please ensure the dataset file exists.")
            sys.exit(1)
        except pd.errors.ParserError:
            print(f"Error parsing CSV file: {path}. Please ensure the file is in correct CSV format.")
            sys.exit(1)
    return datasets, dataset_paths


def correct_str(str_arr):
    """Converts a string representation of a numpy array into a comma-separated string."""
    val_to_ret = (str_arr.replace("[array(", "")
                        .replace("dtype=float32)]", "")
                        .replace("\n","")
                        .replace(" ","")
                        .replace("],","]")
                        .replace("[","")
                        .replace("]",""))
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
    """Extract embeddings and labels"""
    embeddings = np.array([
        np.fromstring(correct_str(embedding), sep=',') for embedding in dataset['embeddings'].tolist()
    ])
    labels = dataset['label'].values
    return embeddings, labels


def train_model(model, train_embeddings, train_labels, val_embeddings, val_labels, epochs=5, batch_size=32, learning_rate=0.001):
    """Train the model and evaluate on validation set after each epoch."""
    # Prepare training data
    train_embeddings_cpu = torch.tensor(train_embeddings, dtype=torch.float32)
    train_labels_cpu = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(train_embeddings_cpu, train_labels_cpu)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare validation data
    val_embeddings_cpu = torch.tensor(val_embeddings, dtype=torch.float32)
    val_labels_cpu = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1)

    val_dataset = TensorDataset(val_embeddings_cpu, val_labels_cpu)
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
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}', end='')

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
        print(f', Validation Accuracy: {val_accuracy:.4f}')

        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = deepcopy(model.state_dict())

    # Load the best model weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, test_embeddings, test_labels, batch_size=32):
    """Evaluate the model and return predictions and probabilities."""
    test_embeddings_cpu = torch.tensor(test_embeddings, dtype=torch.float32)
    test_labels_cpu = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(test_embeddings_cpu, test_labels_cpu)
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
        logging.basicConfig(filename='classification.log', level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return
    logger.info("Execution started.")

    # Load config
    config_parameters = load_config("config.json")

    # Argument parsing
    parser = argparse.ArgumentParser(description="Train probes on processed and labeled datasets.")
    parser.add_argument("--model", help="Model name.")
    parser.add_argument("--layers", nargs='*', help="Layers for embeddings.")
    parser.add_argument("--dataset_names", nargs='*', help="Dataset names.")
    parser.add_argument("--remove_period", action='store_true', help="Remove final period from sentences.")
    parser.add_argument("--save_probes", action='store_true', help="Save trained probes.")
    args = parser.parse_args()
    
    model_name = args.model if args.model is not None else config_parameters["model"]
    remove_period = args.remove_period if args.remove_period else config_parameters["remove_period"]
    layers_to_process = [int(x) for x in args.layers] if args.layers is not None else config_parameters["layers_to_use"]
    dataset_names = args.dataset_names if args.dataset_names is not None else config_parameters["list_of_datasets"]
    save_probes = args.save_probes if args.save_probes else config_parameters["save_probes"]
    input_path = Path(config_parameters["processed_dataset_path"])
    probes_path = Path(config_parameters["probes_dir"])

    for idx in range(len(layers_to_process)):
        # Load datasets
        datasets, dataset_paths = load_datasets(dataset_names, layers_to_process, remove_period, input_path, model_name, idx)

        # Combine all datasets into one DataFrame
        combined_dataset = pd.concat(datasets, ignore_index=True)

        # Split combined_dataset into train, val, test
        train_dataset, temp_dataset = train_test_split(
            combined_dataset,
            test_size=0.2,
            random_state=42,
            stratify=combined_dataset['label']
        )

        val_dataset, test_dataset = train_test_split(
            temp_dataset,
            test_size=0.5,
            random_state=42,
            stratify=temp_dataset['label']
        )

        # Count the number of samples in each split
        num_train_samples = len(train_dataset)
        num_val_samples = len(val_dataset)
        num_test_samples = len(test_dataset)

        print(f"Number of samples in training set: {num_train_samples}")
        print(f"Number of samples in validation set: {num_val_samples}")
        print(f"Number of samples in test set: {num_test_samples}")

        # Display label distribution in each split
        train_label_counts = train_dataset['label'].value_counts().to_dict()
        val_label_counts = val_dataset['label'].value_counts().to_dict()
        test_label_counts = test_dataset['label'].value_counts().to_dict()

        print(f"Label distribution in training set: {train_label_counts}")
        print(f"Label distribution in validation set: {val_label_counts}")
        print(f"Label distribution in test set: {test_label_counts}")

        # Extract embeddings and labels
        def extract_embeddings_and_labels(dataset):
            embeddings = np.array([
                np.fromstring(correct_str(embedding), sep=',') for embedding in dataset['embeddings'].tolist()
            ])
            labels = dataset['label'].values
            return embeddings, labels

        train_embeddings, train_labels = extract_embeddings_and_labels(train_dataset)
        val_embeddings, val_labels = extract_embeddings_and_labels(val_dataset)
        test_embeddings, test_labels = extract_embeddings_and_labels(test_dataset)

        # Train the model    
        model = SAPLMAClassifier(input_dim=train_embeddings.shape[1]).to(device)
        model = train_model(model, train_embeddings, train_labels, val_embeddings, val_labels)
        
        # Evaluate on validation set to find optimal threshold
        val_loss, val_probs, val_true = evaluate_model(model, val_embeddings, val_labels)
        val_probs_flat = val_probs.ravel()
        val_true_flat = val_true.ravel()

        # Compute ROC AUC on validation set
        val_roc_auc, val_fpr, val_tpr, val_thresholds = compute_roc_auc(val_true_flat, val_probs_flat)

        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(val_fpr, val_tpr, val_thresholds)
        
        # Evaluate on test set
        test_loss, test_probs, test_true = evaluate_model(model, test_embeddings, test_labels)
        test_probs_flat = test_probs.ravel()
        test_true_flat = test_true.ravel()

        # Calculate accuracy with optimal threshold
        test_pred_labels = (test_probs_flat >= optimal_threshold).astype(int)
        test_accuracy = accuracy_score(test_true_flat, test_pred_labels)
        test_roc_auc, _, _, _ = compute_roc_auc(test_true_flat, test_probs_flat)

        # Print results
        print(f"Layer {layers_to_process[idx]}: Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_roc_auc:.4f}, Optimal Threshold: {optimal_threshold:.4f}")

        # Save the trained model
        if save_probes:
            os.makedirs(probes_path, exist_ok=True)
            model_path = probes_path / f"{model_name}_{abs(layers_to_process[idx])}_combined.pt"
            torch.save(model.state_dict(), model_path)

        # Save test predictions and probabilities
        test_dataset_copy = test_dataset.copy()
        
        # Remove the 'embeddings' column
        if 'embeddings' in test_dataset_copy.columns:
            test_dataset_copy = test_dataset_copy.drop(columns=['embeddings'])
    
        test_dataset_copy['probability'] = test_probs_flat
        test_dataset_copy['prediction'] = test_pred_labels
        test_dataset_copy['probability'] = test_dataset_copy['probability'].round(4)

        # Save to CSV
        output_path = input_path / f"embeddings_combined{model_name}_{abs(layers_to_process[idx])}_predictions.csv"
        test_dataset_copy.to_csv(output_path, index=False)

    logger.info("Execution completed.")


if __name__ == "__main__":
    main()