#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:45:09 2024

@author: paveenhuang
"""


import os
import sys
from pathlib import Path
import logging
import json
import argparse
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_curve, auc, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import SAPLMAWithCNNRes, AttentionMLP, AttentionMLPSE, AttentionMLPSE1DCNN
from utils import load_config, get_free_gpu, load_data

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

def verify_npy_file(npy_file_path, logger=None):
    try:
        logger.info(f"Verifying .npy file at {npy_file_path}")
        with open(npy_file_path, 'rb') as f:
            magic = f.read(6)
            if magic != b'\x93NUMPY':
                raise ValueError("Not a valid .npy file")
        embeddings = np.load(npy_file_path, mmap_mode='r')
        logger.info(f".npy file is valid. Shape: {embeddings.shape}")
        return True
    except Exception as e:
        logger.error(f"Verification failed for {npy_file_path}: {e}")
        return False

def load_embeddings(npy_file_path, logger):
    try:
        logger.info(f"Starting to load embeddings from {npy_file_path}")
        start_time = time.time()
        embeddings = np.load(npy_file_path, mmap_mode='r')
        load_time = time.time() - start_time
        logger.info(f"Loaded embeddings from {npy_file_path} in {load_time:.2f} seconds. Shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings from {npy_file_path}: {e}")
        return None

def load_datasets_and_embds(dataset_path, dataset_names, remove_period, true_false, embds_path, sanitized_model_name, logger):
    """
    Load datasets and their corresponding embeddings using preallocated memory.
    """
    datasets = []
    embeddings_list = []
    total_samples = 0
    dtype = None  # To store the dtype of embeddings

    for dataset_name in dataset_names: 
        df = load_data(dataset_path, dataset_name, true_false)
        npy_file = embds_path / f"embeddings_{dataset_name}_{sanitized_model_name}.npy"
        embeddings = load_embeddings(npy_file, logger)
        
        if embeddings is None:
            logger.error(f"Failed to load embeddings for dataset {dataset_name}. Skipping.")
            continue
        
        if len(df) != embeddings.shape[0]:
            logger.error(f"Number of embeddings ({embeddings.shape[0]}) does not match number of rows in {dataset_name} ({len(df)}). Skipping.")
            continue
        
        if dtype is None:
            dtype = embeddings.dtype
            logger.info(f"Using dtype {dtype} for combined embeddings.")
        elif dtype != embeddings.dtype:
            logger.error(f"Data type mismatch for dataset {dataset_name}. Expected {dtype}, got {embeddings.dtype}. Skipping.")
            continue
        
        if remove_period:
            df['statement'] = df['statement'].apply(lambda x: x.rstrip(". "))
            
        datasets.append(df)
        embeddings_list.append(embeddings)
        total_samples += embeddings.shape[0]
        logger.info(f"Accumulated {total_samples} samples so far.")

    if not datasets or not embeddings_list:
        logger.error("No valid datasets and embeddings loaded.")
        sys.exit(1)

    # Preallocate combined_embeddings array
    try:
        logger.info(f"Preallocating combined_embeddings array with shape ({total_samples}, {embeddings.shape[1]}, {embeddings.shape[2]}) and dtype {dtype}.")
        start_time = time.time()
        combined_embeddings = np.empty((total_samples, embeddings.shape[1], embeddings.shape[2]), dtype=dtype)
        prealloc_time = time.time() - start_time
        logger.info(f"Preallocated memory in {prealloc_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error preallocating combined_embeddings array: {e}")
        sys.exit(1)
    
    # Second pass: Copy embeddings into preallocated array with progress bar
    start_idx = 0
    for embeddings in tqdm(embeddings_list, desc="Copying embeddings"):
        end_idx = start_idx + embeddings.shape[0]
        logger.info(f"Copying embeddings from index {start_idx} to {end_idx}.")
        copy_start_time = time.time()
        combined_embeddings[start_idx:end_idx] = embeddings[:]
        copy_time = time.time() - copy_start_time
        logger.info(f"Copied embeddings from {start_idx} to {end_idx} in {copy_time:.2f} seconds.")
        start_idx = end_idx

    # Concatenate datasets
    try:
        combined_dataset = pd.concat(datasets, ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_dataset)} samples.")
        logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
    except Exception as e:
        logger.error(f"Error concatenating datasets: {e}")
        sys.exit(1)
        
    return combined_dataset, combined_embeddings

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

def train_layers(
    model, 
    train_embeddings, 
    train_labels, 
    val_embeddings, 
    val_labels, 
    device, 
    epochs=50, 
    batch_size=32, 
    learning_rate=0.005, 
    early_stopping_patience=5
):
    """
    Train the model with Early Stopping based on validation loss, gradient clipping, and learning rate scheduler.

    Parameters:
    - model: The PyTorch model to train.
    - train_embeddings: Training embeddings.
    - train_labels: Training labels.
    - val_embeddings: Validation embeddings.
    - val_labels: Validation labels.
    - device: Device to run the model on.
    - epochs: Maximum number of training epochs.
    - batch_size: Batch size for training.
    - learning_rate: Initial learning rate for the optimizer.
    - early_stopping_patience: Number of epochs to wait for improvement before stopping.
    """
    # Prepare training data
    train_embeddings_tensor = torch.from_numpy(train_embeddings).float()
    train_labels_tensor = torch.from_numpy(train_labels).float().unsqueeze(1)

    train_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # Prepare validation data
    val_embeddings_tensor = torch.from_numpy(val_embeddings).float()
    val_labels_tensor = torch.from_numpy(val_labels).float().unsqueeze(1)

    val_dataset = TensorDataset(val_embeddings_tensor, val_labels_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  
        factor=0.5, 
        patience=2, 
        verbose=True, 
        min_lr=1e-6
    )

    model.to(device)

    best_val_loss = float('inf') 
    best_model_state = None
    early_stopping_counter = 0  

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train() 
        for batch_embeddings, batch_labels in train_dataloader:
            batch_embeddings, batch_labels = batch_embeddings.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            # Added gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}", end="")

        model.eval()  
        all_val_outputs = []
        all_val_labels = []
        val_loss_total = 0.0
        with torch.no_grad():
            for val_embeddings_batch, val_labels_batch in val_dataloader:
                val_embeddings_batch = val_embeddings_batch.to(device, non_blocking=True)
                val_labels_batch = val_labels_batch.to(device, non_blocking=True)
                outputs = model(val_embeddings_batch)
                loss = criterion(outputs, val_labels_batch)
                val_loss_total += loss.item()
                all_val_outputs.append(outputs.cpu())
                all_val_labels.append(val_labels_batch.cpu())

        avg_val_loss = val_loss_total / len(val_dataloader)
        val_outputs_cat = torch.cat(all_val_outputs)
        val_labels_cat = torch.cat(all_val_labels)
        
        # Updated learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Apply sigmoid to get probabilities
        val_probs = torch.sigmoid(val_outputs_cat).numpy()
        val_labels_np = val_labels_cat.numpy()
        
        val_preds = (val_probs >= 0.5).astype(int)
        val_accuracy = accuracy_score(val_labels_np, val_preds)
        print(f", Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        logging.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = deepcopy(model.state_dict())
            early_stopping_counter = 0  
            logging.info(f"New best validation loss: {best_val_loss:.4f} at epoch {epoch + 1}")
        else:
            early_stopping_counter += 1
            logging.info(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")
            if early_stopping_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def evaluate_model(model, test_embeddings, test_labels, device, batch_size=32):
    """Evaluate the model and return predictions and probabilities."""
    test_embeddings_tensor = torch.from_numpy(test_embeddings).float()
    test_labels_tensor = torch.from_numpy(test_labels).float().unsqueeze(1)

    dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
    model.eval()
    model.to(device)
    total_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings, batch_labels = batch_embeddings.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_labels.append(batch_labels.cpu())

    avg_loss = total_loss / len(dataloader)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    return avg_loss, all_outputs, all_labels  # Return raw outputs (logits)

def main():
    # Set up logging
    logger = logging.getLogger(__name__)
    print("Execution started.")
    logger.info("Execution started.")

    # Load config
    config_parameters = load_config()

    # Argument parsing
    parser = argparse.ArgumentParser(description="Train probes on processed and labeled datasets.")
    parser.add_argument("--model", help="Model name.")
    parser.add_argument("--true_false", type=bool, help="Append 'true_false' to dataset name?")
    parser.add_argument("--dataset_names", nargs="*", help="Dataset names.")
    parser.add_argument("--remove_period", action="store_true", help="Remove final period from sentences.")
    parser.add_argument("--save_probes", action="store_true", help="Save trained probes.")
    parser.add_argument("--probe_name", help="Probe name")
    parser.add_argument("--add_name", help="Added probe name")
    
    args = parser.parse_args()
    
    model_name = args.model or config_parameters.get("model")
    probe_name = args.probe_name or config_parameters.get("probe_name")
    add_name = args.add_name or config_parameters.get("add_name")
    remove_period = args.remove_period if args.remove_period else config_parameters.get("remove_period", False)
    true_false = args.true_false if args.true_false is not None else config_parameters.get("true_false", False)
    dataset_names = args.dataset_names if args.dataset_names is not None else config_parameters.get("list_of_datasets", [])
    save_probes = args.save_probes if args.save_probes else config_parameters.get("save_probes", False)
    embds_path = Path(config_parameters.get("processed_dataset_path", "embeddings"))
    dataset_path = Path(config_parameters.get("dataset_path", "datasets"))
    probes_path = Path(config_parameters.get("probes_dir", "probes"))

    sanitized_model_name = model_name.replace("/", "_")

    # Verify and load embeddings
    valid_dataset_names = []
    for dataset_name in dataset_names:
        npy_file = embds_path / f"embeddings_{dataset_name}_{sanitized_model_name}.npy"
        if verify_npy_file(npy_file, logger):
            valid_dataset_names.append(dataset_name)
        else:
            logger.error(f".npy file for {dataset_name} is invalid. Skipping.")
    
    if not valid_dataset_names:
        logger.error("No valid datasets to process. Exiting.")
        sys.exit(1)

    # Load datasets and embeddings
    combined_dataset, combined_embeddings = load_datasets_and_embds(
        dataset_path=dataset_path,
        dataset_names=valid_dataset_names,
        true_false=true_false,
        remove_period=remove_period,
        embds_path=embds_path,
        sanitized_model_name=sanitized_model_name,
        logger=logger
    )
    
    # Make sure combined_embeddings is in contiguous memory layout
    if not combined_embeddings.flags['C_CONTIGUOUS']:
        combined_embeddings = np.ascontiguousarray(combined_embeddings)
        logger.info("Converted combined_embeddings to C-contiguous array.")
        print("Converted combined_embeddings to C-contiguous array.")

    logger.info("Starting manual data splitting.")
    print("Starting manual data splitting.")

    # split datasets
    split_start_time = time.time()
    shuffled_indices = np.random.permutation(len(combined_dataset))
    train_frac = 0.8
    val_frac = 0.1

    train_end = int(train_frac * len(shuffled_indices))
    val_end = train_end + int(val_frac * len(shuffled_indices))

    train_indices = shuffled_indices[:train_end]
    val_indices = shuffled_indices[train_end:val_end]
    test_indices = shuffled_indices[val_end:]

    train_dataset = combined_dataset.iloc[train_indices].reset_index(drop=True)
    val_dataset = combined_dataset.iloc[val_indices].reset_index(drop=True)
    test_dataset = combined_dataset.iloc[test_indices].reset_index(drop=True)

    train_embeddings = combined_embeddings[train_indices]
    val_embeddings = combined_embeddings[val_indices]
    test_embeddings = combined_embeddings[test_indices]

    split_time = time.time() - split_start_time
    logger.info(f"Data splitting completed in {split_time:.2f} seconds.")
    print(f"Data splitting completed in {split_time:.2f} seconds.")
    
    # labels
    train_labels = train_dataset["label"].values
    val_labels = val_dataset["label"].values
    test_labels = test_dataset["label"].values

    # Count the number of samples in each split
    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    num_test_samples = len(test_dataset)

    print(f"Number of samples in training set: {num_train_samples}")
    print(f"Number of samples in validation set: {num_val_samples}")
    print(f"Number of samples in test set: {num_test_samples}")
    logger.info(f"Number of samples in training set: {num_train_samples}")
    logger.info(f"Number of samples in validation set: {num_val_samples}")
    logger.info(f"Number of samples in test set: {num_test_samples}")

    # Display label distribution in each split
    train_label_counts = train_dataset["label"].value_counts().to_dict()
    val_label_counts = val_dataset["label"].value_counts().to_dict()
    test_label_counts = test_dataset["label"].value_counts().to_dict()

    print(f"Label distribution in training set: {train_label_counts}")
    print(f"Label distribution in validation set: {val_label_counts}")
    print(f"Label distribution in test set: {test_label_counts}")
    logger.info(f"Label distribution in training set: {train_label_counts}")
    logger.info(f"Label distribution in validation set: {val_label_counts}")
    logger.info(f"Label distribution in test set: {test_label_counts}")

    # Train the model
    num_layers = train_embeddings.shape[1]
    hidden_size = train_embeddings.shape[2]
    
    print(f"num_layers {num_layers}, hidden_size {hidden_size}")
    logger.info(f"num_layers {num_layers}, hidden_size {hidden_size}")

    # model = SAPLMAWithCNN(hidden_dim=hidden_size, num_layers=num_layers).to(device)
    # model = SAPLMAWithCNNRes(hidden_dim=hidden_size, num_layers=num_layers).to(device)
    # model = AttentionMLP(hidden_size=hidden_size, num_layers=num_layers, num_heads=8, dropout=0.1).to(device)
    # model = AttentionMLPSE(hidden_size=hidden_size, num_layers=num_layers, num_heads=8, dropout=0.1).to(device)
    model = AttentionMLPSE1DCNN(hidden_size=hidden_size, num_layers=num_layers, num_heads=8, dropout=0.1, reduction=16).to(device)
    
    model = train_layers(
        model, 
        train_embeddings, 
        train_labels, 
        val_embeddings, 
        val_labels, 
        device, 
        epochs=20,  
        batch_size=32, 
        learning_rate=0.005, 
        early_stopping_patience=5  
        )
    
    # Evaluate on validation set to find optimal threshold
    val_loss, val_outputs, val_true = evaluate_model(model, val_embeddings, val_labels, device)
    val_probs = torch.sigmoid(val_outputs).numpy().ravel()
    val_true_flat = val_true.numpy().ravel()

    # Compute ROC AUC on validation set
    val_roc_auc, val_fpr, val_tpr, val_thresholds = compute_roc_auc(val_true_flat, val_probs)
    logger.info(f"Validation ROC AUC: {val_roc_auc:.4f}")

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(val_fpr, val_tpr, val_thresholds)
    logger.info(f"Optimal threshold found: {optimal_threshold:.4f}")

    # Convert optimal_threshold to native Python float
    optimal_threshold_value = float(optimal_threshold)

    # Save the optimal threshold to threshold.json
    probe_name_full = f"{sanitized_model_name}_{probe_name}"
    threshold_file = "threshold.json"
    save_threshold(threshold_file, probe_name_full, optimal_threshold_value)

    # Evaluate on test set
    test_loss, test_outputs, test_true = evaluate_model(model, test_embeddings, test_labels, device)
    test_probs = torch.sigmoid(test_outputs).numpy().ravel()
    test_true_flat = test_true.numpy().ravel()

    # Calculate accuracy with optimal threshold
    test_pred_labels = (test_probs >= optimal_threshold_value).astype(int)
    test_accuracy = accuracy_score(test_true_flat, test_pred_labels)
    test_roc_auc, _, _, _ = compute_roc_auc(test_true_flat, test_probs)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_roc_auc:.4f}, Optimal Threshold: {optimal_threshold_value:.4f}")

    # Print results
    print(f"Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_roc_auc:.4f}, Optimal Threshold: {optimal_threshold_value:.4f}")

    # Save the trained model
    if save_probes:
        os.makedirs(probes_path, exist_ok=True)
        probe_path = probes_path / f"{probe_name_full}_{add_name}.pt"
        torch.save(model.state_dict(), probe_path)
        logger.info(f"Trained model saved to {probe_path}")

    # Save test predictions and probabilities
    test_dataset_copy = test_dataset.copy()

    # Remove the 'embeddings' column if it exists
    if "embeddings" in test_dataset_copy.columns:
        test_dataset_copy = test_dataset_copy.drop(columns=["embeddings"])

    test_dataset_copy["probability"] = test_probs
    test_dataset_copy["prediction"] = test_pred_labels
    test_dataset_copy["probability"] = test_dataset_copy["probability"].round(4)

    # Save to CSV
    output_dir = os.path.join(os.getcwd(), "prediction")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{probe_name_full}_predictions.csv")
    test_dataset_copy.to_csv(output_path, index=False)

    logger.info("Execution completed.")
    print("Execution completed.")

if __name__ == "__main__":
    os.environ["HF_HOME"] = "/data2/cache/d12922004"

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="embedding_extraction.log",
        filemode='a'
    )
    logger = logging.getLogger(__name__)

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()

    # Select device based on availability
    if cuda_available:
        device_id = get_free_gpu(cuda_available)
        if device_id is not None:
            device = torch.device(f"cuda:{device_id}")
            logger.info(f"Using GPU: cuda:{device_id}")
        else:
            device = torch.device("cuda:0")
            logger.info("Falling back to GPU: cuda:0")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # If available, choose Apple's MPS
            logger.info("Using MPS device")
        else:
            device = torch.device("cpu")  # If no GPU and MPS, use CPU
            logger.info("Using CPU")

    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    main()