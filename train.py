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
from copy import deepcopy
import sys
import argparse

from model import SAPLMAClassifier


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


def prepare_datasets(datasets, dataset_names, test_first_only):
    """Prepare train and test datasets."""
    if not datasets or not dataset_names:
        raise ValueError("Both 'datasets' and 'dataset_names' must be nonempty.")
    if len(datasets) != len(dataset_names):
        raise ValueError("'datasets' and 'dataset_names' must have the same length.")

    train_datasets, test_datasets = [], []
    dataset_loop_length = 1 if test_first_only else len(dataset_names)
    for ds in range(dataset_loop_length):
        if test_first_only:
            test_df = datasets[0]
            train_df = pd.concat(datasets[1:], ignore_index=True)
        else:
            test_df = datasets[ds]
            train_df = pd.concat(datasets[:ds] + datasets[ds + 1:], ignore_index=True)
        train_datasets.append(train_df)
        test_datasets.append(test_df)
    return train_datasets, test_datasets


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


def print_results(results, dataset_names, repeat_each, layer_num_from_end):
    """Prints average accuracy, AUC, and optimal threshold for each dataset."""
    if len(results) != len(dataset_names) * repeat_each:
        raise ValueError("Results array length should be equal to dataset_names length multiplied by repeat_each.")
    overall_res = []
    for ds in range(len(dataset_names)):
        relevant_results_portion = results[repeat_each*ds:repeat_each*(ds+1)]
        acc_list = [t[2] for t in relevant_results_portion]
        auc_list = [t[3] for t in relevant_results_portion]
        opt_thresh_list = [t[4] for t in relevant_results_portion]
        avg_acc = sum(acc_list) / len(acc_list)
        avg_auc = sum(auc_list) / len(auc_list)
        avg_thrsh = sum(opt_thresh_list) / len(opt_thresh_list)
        text_res = (f"Dataset: {dataset_names[ds]}, Layer: {layer_num_from_end}, "
                    f"Avg Accuracy: {avg_acc:.4f}, Avg AUC: {avg_auc:.4f}, "
                    f"Avg Threshold: {avg_thrsh:.4f}")
        print(text_res)
        overall_res.append(text_res)
    return overall_res


def train_model(model, train_embeddings, train_labels, epochs=5, batch_size=32, learning_rate=0.001):
    """Train the model."""
    train_embeddings_cpu = torch.tensor(train_embeddings, dtype=torch.float32)
    train_labels_cpu = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(train_embeddings_cpu, train_labels_cpu)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}')
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
    parser.add_argument("--remove_period", type=bool, help="Remove final period from sentences.")
    parser.add_argument("--test_first_only", type=bool, help="Use only the first dataset for testing.")
    parser.add_argument("--save_probes", type=bool, help="Save trained probes.")
    parser.add_argument("--repeat_each", type=int, default=1, help="Repeat training.")
    args = parser.parse_args()
    
    model_name = args.model if args.model is not None else config_parameters["model"]
    remove_period = args.remove_period if args.remove_period is not None else config_parameters["remove_period"]
    layers_to_process = [int(x) for x in args.layers] if args.layers is not None else config_parameters["layers_to_use"]
    dataset_names = args.dataset_names if args.dataset_names is not None else config_parameters["list_of_datasets"]
    test_first_only = args.test_first_only if args.test_first_only is not None else config_parameters["test_first_only"]
    save_probes = args.save_probes if args.save_probes is not None else config_parameters["save_probes"]
    repeat_each = args.repeat_each if args.repeat_each is not None else config_parameters["repeat_each"]
    input_path = Path(config_parameters["processed_dataset_path"])
    probes_path = Path(config_parameters["probes_dir"])

    overall_res = []

    for idx in range(len(layers_to_process)):
        results = []
        datasets, dataset_paths = load_datasets(dataset_names, layers_to_process, remove_period, input_path, model_name, idx)
        train_datasets, test_datasets = prepare_datasets(datasets, dataset_names, test_first_only)

        for count, (train_dataset, test_dataset, test_dataset_path) in enumerate(zip(train_datasets, test_datasets, dataset_paths)):
            # Split the dataset into embeddings and labels
            train_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in train_dataset['embeddings'].tolist()])
            train_labels = train_dataset['label']
            test_embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in test_dataset['embeddings'].tolist()])
            test_labels = test_dataset['label']

            best_accuracy = 0
            best_model = None
            all_probs_list = []

            for i in range(repeat_each):
                model = SAPLMAClassifier(input_dim=train_embeddings.shape[1]).to(device)
                model = train_model(model, train_embeddings, train_labels)

                # Evaluate the model
                loss, test_probs, test_true = evaluate_model(model, test_embeddings, test_labels)
                test_probs_flat = test_probs.ravel()
                test_true_flat = test_true.ravel()

                # Compute ROC AUC
                roc_auc, fpr, tpr, thresholds = compute_roc_auc(test_true_flat, test_probs_flat)

                # Find optimal threshold
                optimal_threshold = find_optimal_threshold(fpr, tpr, thresholds)

                # Calculate accuracy with optimal threshold
                test_pred_labels = (test_probs_flat >= optimal_threshold).astype(int)
                accuracy = accuracy_score(test_true_flat, test_pred_labels)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = deepcopy(model)

                all_probs_list.append(test_probs_flat)

                # Store results for printing
                results.append((dataset_names[count], i, accuracy, roc_auc, optimal_threshold))

            # Save the best model
            if save_probes and best_model is not None:
                os.makedirs(probes_path, exist_ok=True)
                model_path = probes_path / f"{model_name}_{layers_to_process[idx]}_{dataset_names[count]}.pt"
                torch.save(best_model.state_dict(), model_path)

            # Save test predictions and probabilities
            test_dataset_copy = test_dataset.copy()
            test_dataset_copy['average_probability'] = np.mean(all_probs_list, axis=0)
            for i, probs in enumerate(all_probs_list):
                test_dataset_copy[f'model_{i}_probability'] = probs
            # Round probabilities
            prob_cols = [col for col in test_dataset_copy.columns if 'probability' in col]
            test_dataset_copy[prob_cols] = test_dataset_copy[prob_cols].round(4)

            # Save to CSV
            original_path = Path(test_dataset_path)
            new_path = original_path.with_name(original_path.name.replace('.csv', '_predictions.csv'))
            test_dataset_copy.to_csv(new_path, index=False)

        # Print and log results
        avg_res = print_results(results, dataset_names, repeat_each, layers_to_process[idx])
        overall_res.extend(avg_res)

    logger.info("Execution completed.")
    logger.info("Overall results: " + str(overall_res))


if __name__ == "__main__":
    main()