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

        # CNN 
        self.conv1 = nn.Conv1d(in_channels=num_layers, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Reduce spatial dimension by 2

        # FC
        # After three conv layers and pooling, spatial dimension is 32 / 2 / 2 / 2 = 4
        # Thus, x shape: batch_size x 16 x 4
        self.fc1 = nn.Linear(16 * 4, 1024)  # 16 * 4 = 64
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

        # 激活函数和正则化
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: batch_size × num_layers × hidden_dim
        # 交换维度以匹配卷积层输入格式
        x = x.permute(0, 2, 1)  # 现在 x 的形状是 batch_size × hidden_dim × num_layers

        # 卷积层
        x = self.conv1(x)  # batch_size x 64 x 4096
        x = self.leaky_relu(x)
        x = self.max_pool(x)  # batch_size x 64 x 2048

        x = self.conv2(x)  # batch_size x 32 x 2048
        x = self.leaky_relu(x)
        x = self.max_pool(x)  # batch_size x 32 x 1024

        x = self.conv3(x)  # batch_size x 16 x 1024
        x = self.leaky_relu(x)
        x = self.max_pool(x)  # batch_size x 16 x 512

        # 再次应用池化以进一步减少空间维度
        x = self.max_pool(x)  # batch_size x 16 x 256
        x = self.max_pool(x)  # batch_size x 16 x 128
        x = self.max_pool(x)  # batch_size x 16 x 64
        x = self.max_pool(x)  # batch_size x 16 x 32
        x = self.max_pool(x)  # batch_size x 16 x 16
        x = self.max_pool(x)  # batch_size x 16 x 8
        x = self.max_pool(x)  # batch_size x 16 x 4

        # 展平
        x = x.view(x.size(0), -1)  # batch_size x (16 * 4) = batch_size x 64

        # 全连接层
        x = self.fc1(x)  # 64 -> 1024
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)  # 1024 -> 512
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc3(x)  # 512 -> 256
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc4(x)  # 256 -> 1
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