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
    def __init__(self, hidden_dim, num_layers, dropout):
        super(SAPLMAWithCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1DCNN
        self.conv1 = nn.Conv1d(in_channels=num_layers, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        # MLP
        self.fc1 = nn.Linear(hidden_dim, 1024)  # 4096 -> 1024
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)         # 1024 -> 512
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)          # 512 -> 256
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)            # 256 -> 1

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # CNN
        x = self.conv1(x)  # [batch_size, 16, hidden_dim]
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)  # [batch_size, 1, hidden_dim]
        x = self.leaky_relu(x)

        # Flatten: [batch_size, hidden_dim]
        x = x.view(x.size(0), -1)

        # FC
        x = self.fc1(x)          # [batch_size, 1024]
        x = self.bn_fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)          # [batch_size, 512]
        x = self.bn_fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc3(x)          # [batch_size, 256]
        x = self.bn_fc3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        logits = self.fc4(x)     # [batch_size, 1]
        return logits             

    
class SAPLMAWithCNNRes(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout):
        super(SAPLMAWithCNNRes, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        # CNN
        self.conv1 = nn.Conv1d(in_channels=num_layers, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels=num_layers, out_channels=1, kernel_size=1)
        self.residual_bn = nn.BatchNorm1d(1)

        # MLP
        self.fc1 = nn.Linear(self.hidden_dim, 1024)  # 4096 -> 1024
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)         # 1024 -> 512
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)          # 512 -> 256
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)            # 256 -> 1

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward propagation function.
        Parameters: - x: input tensor, shape is [batch_size, num_layers=32, hidden_dim=4096]
        Returns: - logits: prediction results without Sigmoid activation, shape is [batch_size, 1]
        """
        # CNN
        x_main = self.conv1(x)      # [batch_size, 16, 4096]
        x_main = self.bn1(x_main)
        x_main = self.leaky_relu(x_main)

        x_main = self.conv2(x_main) # [batch_size, 1, 4096]
        x_main = self.bn2(x_main)
        x_main = self.leaky_relu(x_main)

        # Residual connection
        x_residual = self.residual_conv(x)   # [batch_size, 1, 4096]
        x_residual = self.residual_bn(x_residual)

        # Adding residuals
        x = x_main + x_residual              # [batch_size, 1, 4096]
        x = self.leaky_relu(x)

        # Flatten
        x = x.view(x.size(0), -1)            # [batch_size, 4096]

        # FC
        x = self.fc1(x)                       # [batch_size, 1024]
        x = self.bn_fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)                       # [batch_size, 512]
        x = self.bn_fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc3(x)                       # [batch_size, 256]
        x = self.bn_fc3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        logits = self.fc4(x)                  # [batch_size, 1]
        return logits
    

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
        
        # FC layers 
        self.fc1 = nn.Linear(hidden_size, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.fc4 = nn.Linear(256, 1)        
        
        # BN
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3_bn = nn.BatchNorm1d(256)
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
                
        return out
    

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [batch_size, channel]
        se = self.fc1(x)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se

class AttentionMLPSE(nn.Module):
    """
    Aggregate multiple layers of embeddings and perform classification using the attention mechanism
    Enhanced with Layer Normalization and SE blocks for improved performance and stability
    """
    def __init__(self, hidden_size=4096, num_layers=32, num_heads=8, dropout=0.1):
        super(AttentionMLPSE, self).__init__()
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.dropout_attention = nn.Dropout(p=dropout)
        
        # SE Blocks integrated into MLP
        self.fc1 = nn.Linear(hidden_size, 1024)
        self.se1 = SEBlock(1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(1024, 512)
        self.se2 = SEBlock(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.fc3 = nn.Linear(512, 256)
        self.se3 = SEBlock(256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.fc4 = nn.Linear(256, 1)
        
        # BN
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4_bn = nn.BatchNorm1d(1)
        
    def forward(self, x):
        """
        x: Tensor (batch_size, num_layers, hidden_size)
        out: Tensor (batch_size, 1)
        """
        # Multi-head Attention with Residual Connection and LayerNorm
        attn_output, attn_weights = self.attention(x, x, x)  # [batch_size, num_layers, hidden_size]
        x = self.attention_norm(attn_output + x)  # Residual Connection
        x = self.dropout_attention(x)
        
        # Global Average Pooling
        pooled_output = x.mean(dim=1)  # [batch_size, hidden_size]
        
        # FC1 with SE
        fc1_out = self.fc1(pooled_output)
        fc1_out = self.fc1_bn(fc1_out)
        fc1_out = self.se1(fc1_out)
        fc1_out = self.relu1(fc1_out)
        fc1_out = self.dropout1(fc1_out)
        
        # FC2 with SE
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.fc2_bn(fc2_out)
        fc2_out = self.se2(fc2_out)
        fc2_out = self.relu2(fc2_out)
        fc2_out = self.dropout2(fc2_out)
        
        # FC3 with SE
        fc3_out = self.fc3(fc2_out)
        fc3_out = self.fc3_bn(fc3_out)
        fc3_out = self.se3(fc3_out)
        fc3_out = self.relu3(fc3_out)
        fc3_out = self.dropout3(fc3_out)
        
        # FC4
        logits = self.fc4(fc3_out)
        logits = self.fc4_bn(logits)  
        
        return logits  

class AttentionMLPSE1DCNN(nn.Module):
    """
    Aggregate multiple layers of embeddings and perform classification using the attention mechanism.
    Enhanced with SE blocks and 1DCNN for improved feature aggregation and stability.
    """
    def __init__(self, hidden_size=4096, num_layers=32, num_heads=8, dropout=0.1, reduction=16):
        super(AttentionMLPSE1DCNN, self).__init__()
        
        # multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.dropout_attention = nn.Dropout(p=dropout)
        
        # 1DCNN
        self.conv1 = nn.Conv1d(in_channels=num_layers, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        
        # Residual Connection
        self.residual_conv = nn.Conv1d(in_channels=num_layers, out_channels=1, kernel_size=1)
        self.residual_bn = nn.BatchNorm1d(1)
        
        # MLP 
        self.fc1 = nn.Linear(hidden_size, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.se1 = SEBlock(1024, reduction)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.se2 = SEBlock(512, reduction)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.se3 = SEBlock(256, reduction)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.fc4 = nn.Linear(256, 1)
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
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Attention
        attn_output, attn_weights = self.attention(x, x, x)  # [32, 32, 4096]
        x = self.attention_norm(attn_output + x)           
        x = self.dropout_attention(x)
        
        # 1DCNN 
        x_main = self.conv1(x)                         # [32, 16, 4096]
        x_main = self.bn1(x_main)
        x_main = self.leaky_relu(x_main)
        
        x_main = self.conv2(x_main)                         # [32, 1, 4096]
        x_main = self.bn2(x_main)
        x_main = self.leaky_relu(x_main)
        
        # RC
        x_residual = self.residual_conv(x)              # [32, 1, 4096]
        x_residual = self.residual_bn(x_residual)
        x_cnn = x_main + x_residual                         # [32, 1, 4096]
        x_cnn = self.leaky_relu(x_cnn)
        
        # flatten
        x_cnn = x_cnn.view(x_cnn.size(0), -1)              # [32, 4096]
        
        # FC1 with SE
        fc1_out = self.fc1(x_cnn)                           # [32, 1024]
        fc1_out = self.fc1_bn(fc1_out)
        fc1_out = self.se1(fc1_out)
        fc1_out = self.relu1(fc1_out)
        fc1_out = self.dropout1(fc1_out)
        
        # FC2 with SE
        fc2_out = self.fc2(fc1_out)                         # [32, 512]
        fc2_out = self.fc2_bn(fc2_out)
        fc2_out = self.se2(fc2_out)
        fc2_out = self.relu2(fc2_out)
        fc2_out = self.dropout2(fc2_out)
        
        # FC3 with SE
        fc3_out = self.fc3(fc2_out)                         # [32, 256]
        fc3_out = self.fc3_bn(fc3_out)
        fc3_out = self.se3(fc3_out)
        fc3_out = self.relu3(fc3_out)
        fc3_out = self.dropout3(fc3_out)
        
        # FC4
        logits = self.fc4(fc3_out)                          # [32, 1]
        logits = self.fc4_bn(logits)
        
        return logits    


class AttentionMLPSE2DCNN(nn.Module):
    """
    Aggregate multiple layers of embeddings and perform classification using the attention mechanism.
    Enhanced with SE blocks and 2DCNN for improved feature aggregation and stability.
    """
    def __init__(self, hidden_size=4096, num_layers=32, num_heads=4, dropout=0.5, reduction=16):
        super(AttentionMLPSE2DCNN, self).__init__()
        
        # Multi-head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.dropout_attention = nn.Dropout(p=dropout)
        
        # 2DCNN
        self.conv1 = nn.Conv2d(
            in_channels=1,               # 单通道输入
            out_channels=16,             # 输出通道数
            kernel_size=(7, 7),          # 卷积核大小
            stride=(1, 1),
            padding=(3, 3)               # 仅在高度维度上填充
        )
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=1,              # 输出通道数为1，实现特征聚合
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )
        self.bn2 = nn.BatchNorm2d(1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        
        # Residual Connection
        self.residual_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1)
        )
        self.residual_bn = nn.BatchNorm2d(1)
        
        # Adaptive Pooling to reduce spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, hidden_size))  # [batch_size, 1, 1, hidden_size]
        
        # MLP 部分集成 SE 块
        self.fc1 = nn.Linear(hidden_size, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.se1 = SEBlock(1024, reduction)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.se2 = SEBlock(512, reduction)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.se3 = SEBlock(256, reduction)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.fc4 = nn.Linear(256, 1)
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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Multi-head Attention
        attn_output, attn_weights = self.attention(x, x, x)  # [batch_size, num_layers, hidden_size]        
        x = self.attention_norm(attn_output + x)            
        x = self.dropout_attention(x)
        
        # 2DCNN 
        x_cnn = x.unsqueeze(1)                              # [batch_size, 1, num_layers, hidden_size]
        
        # Apply 2DCNN
        x_main = self.conv1(x_cnn)                         # [batch_size, 16, num_layers, hidden_size]
        x_main = self.bn1(x_main)
        x_main = self.leaky_relu(x_main)    
        x_main = self.conv2(x_main)                         # [batch_size, 1, num_layers, hidden_size]
        x_main = self.bn2(x_main)
        x_main = self.leaky_relu(x_main)
        
        # Residual Connection
        x_residual = self.residual_conv(x_cnn)              # [batch_size, 1, num_layers, hidden_size]
        x_residual = self.residual_bn(x_residual)        
        x_cnn = x_main + x_residual                         # [batch_size, 1, num_layers, hidden_size]
        x_cnn = self.leaky_relu(x_cnn)
        
        # Adaptive Pooling to reduce spatial dimensions to [1, hidden_size]
        x_cnn = self.adaptive_pool(x_cnn)                    # [batch_size, 1, 1, hidden_size]
        
        # Flatten to [batch_size, hidden_size]
        x_cnn = x_cnn.view(x_cnn.size(0), -1)              # [batch_size, hidden_size]
        
        # FC1 with SE
        fc1_out = self.fc1(x_cnn)                           # [batch_size, 1024]        
        fc1_out = self.fc1_bn(fc1_out)        
        fc1_out = self.se1(fc1_out)        
        fc1_out = self.relu1(fc1_out)        
        fc1_out = self.dropout1(fc1_out)
        
        # FC2 with SE
        fc2_out = self.fc2(fc1_out)                         # [batch_size, 512]        
        fc2_out = self.fc2_bn(fc2_out)        
        fc2_out = self.se2(fc2_out)        
        fc2_out = self.relu2(fc2_out)        
        fc2_out = self.dropout2(fc2_out)
        
        # FC3 with SE
        fc3_out = self.fc3(fc2_out)                         # [batch_size, 256]        
        fc3_out = self.fc3_bn(fc3_out)        
        fc3_out = self.se3(fc3_out)        
        fc3_out = self.relu3(fc3_out)        
        fc3_out = self.dropout3(fc3_out)
        
        # FC4
        logits = self.fc4(fc3_out)                          # [batch_size, 1]        
        logits = self.fc4_bn(logits)
        
        return logits                                        # 返回 logits                                   
    
if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch_size = 32
    num_layers = 32
    hidden_size = 4096
    num_heads = 8
    dropout = 0.1
    reduction = 16
    
    model = AttentionMLPSE2DCNN(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        reduction=reduction
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = torch.randn(batch_size, num_layers, hidden_size).to(device)  # [32, 32, 4096]
    output = model(input_tensor)
    print(f"Final output shape: {output.shape}")  # 应为 [32, 1]