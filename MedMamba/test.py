#!/usr/bin/env python3
"""
Test script for MedMamba
直接在配置文件 configs/config3d.py 中修改参数，然后运行：python test.py
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import Model
from data import create_data_loaders
from utils.metrics import MetricsCalculator


def main():
    
    


if __name__ == '__main__':
    main()

