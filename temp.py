#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:23:28 2024

@author: paveenhuang
"""

from transformers import LlamaForCausalLM

def get_model_layers(model_name: str):
    """
    获取指定 LLaMA 模型的层数。
    """
    try:
        # 加载模型配置
        model = LlamaForCausalLM.from_pretrained(model_name)
        config = model.config
        
        # 打印模型配置中的层数信息
        num_layers = config.num_hidden_layers
        print(f"Model: {model_name}")
        print(f"Number of hidden layers: {num_layers}")
        
        # 如果需要更多详细信息，可以打印整个配置
        # print(config)
        
        return num_layers
    except Exception as e:
        print(f"Error loading model configuration: {e}")

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # 替换为您的模型名称
    get_model_layers(model_name)