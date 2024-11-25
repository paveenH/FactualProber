#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:11:47 2024

@author: paveenhuang
"""

import torch
import torch.nn as nn
    
    
class SAPLMAClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SAPLMAClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out
    

class SAPLMAWithCNN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(SAPLMAWithCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(in_channels=num_layers, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(hidden_dim, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: batch_size × layers × hidden_dim
        batch_size = x.size(0)

        # Apply CNN to reduce layers dimension
        x = self.conv1(x)  # shape: batch_size × 16 × hidden_dim
        x = self.relu1(x)
        x = self.conv2(x)  # shape: batch_size × 1 × hidden_dim
        x = self.relu2(x)

        # Flatten for fully connected layers
        x = x.view(batch_size, -1)  # shape: batch_size × hidden_dim

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.relu5(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


class AttentionMLP(nn.Module):
    """
    Aggregate multiple layers of embeddings and perform classification using the attention mechanism
    Enhanced with Layer Normalization for improved performance and stability
    """
    def __init__(self, hidden_size=4096, num_layers=32, num_heads=8, dropout=0.1):
        super(AttentionMLP, self).__init__()
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # FC layers without residual connections
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.fc4 = nn.Linear(64, 1)        
        self.sigmoid = nn.Sigmoid()
        
        # BN
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3_bn = nn.BatchNorm1d(64)
        self.fc4_bn = nn.BatchNorm1d(1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: Tensor (batch_size, num_layers, hidden_size)
        out: Tensor (batch_size, 1)
        """
        # Multi-head Attention with Residual Connection and LayerNorm
        attn_output, attn_weights = self.attention(x, x, x)  # attn_output: (batch_size, num_layers, hidden_size)
        x = self.attention_norm(attn_output + x)  # Residual Connection (valid shape)
        
        # Compute mean of attention weights
        attn_weights_mean = attn_weights.mean(dim=1) # [32, 32, 32]  [batch_size, num_layers, num_layers]
        attn_weights_normalized = attn_weights_mean / attn_weights_mean.sum(dim=1, keepdim=True)  # [32, 32, 32]
        
        # Mean pooling with normalized attention weights
        pooled_output = torch.bmm(attn_weights_normalized.unsqueeze(1), x).squeeze(1)  
        
        # MLP layers without Residual Connections
        # FC1
        fc1_out = self.fc1(pooled_output)
        fc1_out = self.fc1_bn(fc1_out)  
        fc1_out = self.relu1(fc1_out)
        fc1_out = self.dropout1(fc1_out)

        # FC2
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.fc2_bn(fc2_out)  
        fc2_out = self.relu2(fc2_out)
        fc2_out = self.dropout2(fc2_out)

        # FC3
        fc3_out = self.fc3(fc2_out)
        fc3_out = self.fc3_bn(fc3_out)  
        fc3_out = self.relu3(fc3_out)
        fc3_out = self.dropout3(fc3_out)

        # FC4
        out = self.fc4(fc3_out)
        out = self.fc4_bn(out)  
        
        out = self.sigmoid(out)
        
        return out

class AttentionMLPReduction(nn.Module):
    """
    Aggregate multiple layers of embeddings and perform classification using the attention mechanism
    """
    def __init__(self, hidden_size=4096, reduced_size=2048, num_layers=32, num_heads=8, dropout=0.1):
        super(AttentionMLPReduction, self).__init__()
        # Dimension Reduction Layer
        self.dim_reduction = nn.Linear(hidden_size, reduced_size)
        self.relu_reduction = nn.ReLU()
        self.dropout_reduction = nn.Dropout(p=dropout)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=reduced_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # FC layers
        self.fc1 = nn.Linear(reduced_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
    
    def forward(self, x):
        """
        x: Tensor (batch_size, num_layers, hidden_size)
        out: Tensor (batch_size, 1)
        """
        # Dimension Reduction
        x = self.dim_reduction(x)  # (batch_size, num_layers, reduced_size)
        x = self.relu_reduction(x)
        x = self.dropout_reduction(x)
        
        # Apply attention
        attn_output, attn_weights = self.attention(x, x, x)  # attn_output: (batch_size, num_layers, reduced_size)
        
        # Compute mean attention weights across heads
        # attn_weights shape: (batch_size, num_heads, num_layers, num_layers)
        attn_weights_mean = attn_weights.mean(dim=1)  # (batch_size, num_layers, num_layers)
        
        # Normalize attention weights across the sequence length
        attn_weights_normalized = attn_weights_mean / attn_weights_mean.sum(dim=-1, keepdim=True)  # (batch_size, num_layers, num_layers)
        
        # Weighted sum of embeddings
        # pooled_output = torch.bmm(attn_weights_normalized, attn_output)  # (batch_size, num_layers, reduced_size)
        pooled_output = torch.matmul(attn_weights_normalized, attn_output)
        # pooled_output = (attn_weights_normalized.unsqueeze(-1) * attn_output).sum(dim=2) 
        
        pooled_output = pooled_output.mean(dim=1)  # (batch_size, reduced_size)
        
        # MLP
        out = self.fc1(pooled_output)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        
        return out

class AttentionMLP1DCNN(nn.Module):
    def __init__(self, hidden_size=4096, reduced_size=2048, num_layers=32, num_heads=8, dropout=0.1):
        super(AttentionMLP1DCNN, self).__init__()
        # Convolutional Dimension Reduction
        self.conv1 = nn.Conv1d(in_channels=num_layers, out_channels=num_layers, kernel_size=3, padding=1)
        self.relu_conv = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dim_reduction = nn.Linear(hidden_size, reduced_size)
        self.dropout_reduction = nn.Dropout(p=dropout)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=reduced_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # FC layers
        self.fc1 = nn.Linear(reduced_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
    
    def forward(self, x):
        """
        x: Tensor (batch_size, num_layers, hidden_size)
        out: Tensor (batch_size, 1)
        """
        # Permute for Conv1d: (batch_size, hidden_size, num_layers)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)  # (batch_size, hidden_size, num_layers)
        x = self.relu_conv(x)
        x = self.pool(x)    # (batch_size, hidden_size, num_layers//2)
        x = x.permute(0, 2, 1)  # (batch_size, num_layers//2, hidden_size)
        
        # Dimension Reduction
        x = self.dim_reduction(x)  # (batch_size, num_layers//2, reduced_size)
        x = self.relu_conv(x)
        x = self.dropout_reduction(x)
        
        # Apply attention
        attn_output, attn_weights = self.attention(x, x, x)  # attn_output: (batch_size, num_layers//2, reduced_size)
        
        # Compute mean attention weights across heads
        attn_weights_mean = attn_weights.mean(dim=1)  # (batch_size, num_layers//2, num_layers//2)
        
        # Normalize attention weights across the sequence length
        attn_weights_normalized = attn_weights_mean / attn_weights_mean.sum(dim=-1, keepdim=True)  # (batch_size, num_layers//2, num_layers//2)
        
        # Weighted sum of embeddings
        pooled_output = torch.bmm(attn_weights_normalized, attn_output)  # (batch_size, num_layers//2, reduced_size)
        pooled_output = pooled_output.mean(dim=1)  # (batch_size, reduced_size)
        
        # MLP
        out = self.fc1(pooled_output)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        
        return out