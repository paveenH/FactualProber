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


class AttentionMLP(nn.Module):
    """
    Aggregate multiple layers of embeddings and perform classification using the attention mechanism
    """
    def __init__(self, hidden_size=1024, num_layers=24, num_heads=8, dropout=0.1):
        super(AttentionMLP, self).__init__()
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        # FC layers
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
        attn_output, attn_weights = self.attention(x, x, x)  # attn_output: (batch_size, num_layers, hidden_size)
        
        attn_weights_mean = attn_weights.mean(dim=1)  # (batch_size, num_layers)
        attn_weights_normalized = attn_weights_mean / attn_weights_mean.sum(dim=1, keepdim=True)  # normalization
        
        # mean pooling
        pooled_output = torch.bmm(attn_weights_normalized.unsqueeze(1), attn_output).squeeze(1)  # (batch_size, hidden_size)
        
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