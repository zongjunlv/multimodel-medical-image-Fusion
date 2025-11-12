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

from models import (
    create_medmamba_tiny, create_medmamba_small, create_medmamba_base,
    create_medmamba3d_tiny, create_medmamba3d_small, create_medmamba3d_base, create_medmamba3d_large
)
from src.data import create_data_loaders
from src.utils.metrics import MetricsCalculator
from configs.config3d import get_config3d, get_medmamba3d_tiny_config

# ============================================================
# 配置区域 - 在这里修改您的测试参数
# ============================================================
USE_CONFIG = "tiny"  # 选择配置：'default', 'tiny', 'small', 'base'
SAVE_RESULTS_PATH = None  # 结果保存路径，None表示保存到checkpoint同目录
# ============================================================


def load_config(config_type='default'):
    """Load configuration"""
    if config_type == 'tiny':
        print("🔍 Loading Tiny 3D configuration...")
        config = get_medmamba3d_tiny_config()
    else:
        print("🔍 Loading default 3D configuration...")
        config = get_config3d()
    
    is_3d = True  # 默认使用3D配置
    return config, is_3d


def create_model(model_name, config, num_classes, is_3d, **kwargs):
    """Create model based on configuration"""
    if is_3d:
        print(f"🔍 Creating 3D model: {model_name}")
        model_creators = {
            'tiny': create_medmamba3d_tiny,
            'small': create_medmamba3d_small,
            'base': create_medmamba3d_base,
            'large': create_medmamba3d_large,
        }
        if model_name not in model_creators:
            raise ValueError(f"Unknown 3D model: {model_name}. Available: {list(model_creators.keys())}")
        
        # Filter out 2D-specific parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['depths', 'dims']}
        return model_creators[model_name](num_classes=num_classes, **filtered_kwargs)
    else:
        print(f"🔍 Creating 2D model: {model_name}")
        model_creators = {
            'tiny': create_medmamba_tiny,
            'small': create_medmamba_small,
            'base': create_medmamba_base,
        }
        if model_name not in model_creators:
            raise ValueError(f"Unknown 2D model: {model_name}. Available: {list(model_creators.keys())}")
        return model_creators[model_name](num_classes=num_classes, **kwargs)


def load_model(checkpoint_path, model, device):
    """Load model weights from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📥 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   📊 Checkpoint info: Epoch {checkpoint.get('epoch', 'N/A')}, "
                  f"Val Acc: {checkpoint.get('best_acc', 'N/A'):.2f}%")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    print("✅ Model loaded successfully")
    
    return model


def evaluate_model(model, test_loader, device, num_classes, class_names=None):
    """
    Evaluate model on test data
    
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    model.eval()
    
    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator(num_classes=num_classes, class_names=class_names)
    
    print(f"\n🔄 Running evaluation on {len(test_loader.dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Update metrics
            metrics_calculator.update(outputs, targets)
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {(batch_idx + 1) * test_loader.batch_size}/{len(test_loader.dataset)} samples")
    
    # Compute final metrics
    metrics = metrics_calculator.compute_metrics()
    
    return metrics


def print_results(metrics, class_names=None):
    """Print formatted test results"""
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    # Overall metrics table
    print(f"\n┌{'─'*68}┐")
    print(f"│ {'Metric':<20} │ {'Value':<45} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'Accuracy':<20} │ {metrics['accuracy']*100:>6.2f}% {' '*37} │")
    print(f"│ {'Precision (Macro)':<20} │ {metrics['precision_macro']:>6.4f} {' '*38} │")
    print(f"│ {'Recall (Macro)':<20} │ {metrics['recall_macro']:>6.4f} {' '*38} │")
    print(f"│ {'F1-Score (Macro)':<20} │ {metrics['f1_macro']:>6.4f} {' '*38} │")
    
    # AUC
    if 'auc' in metrics:
        print(f"│ {'AUC':<20} │ {metrics['auc']:>6.4f} {' '*38} │")
    elif 'auc_macro' in metrics:
        print(f"│ {'AUC (Macro)':<20} │ {metrics['auc_macro']:>6.4f} {' '*38} │")
        print(f"│ {'AUC (Weighted)':<20} │ {metrics['auc_weighted']:>6.4f} {' '*38} │")
    
    print(f"└{'─'*68}┘")
    
    # Per-class metrics
    if class_names and len(class_names) <= 10:
        print(f"\nPer-Class Metrics")
        print(f"┌{'─'*68}┐")
        print(f"│ {'Class':<15} │ {'Precision':>10} │ {'Recall':>10} │ {'F1-Score':>10} │ {'Support':>8} │")
        print(f"├{'─'*68}┤")
        
        cm = metrics.get('confusion_matrix', None)
        for i, class_name in enumerate(class_names):
            if f'precision_{class_name}' in metrics:
                support = cm[i].sum() if cm is not None else 0
                
                print(
                    f"│ {class_name:<15} │ "
                    f"{metrics[f'precision_{class_name}']:>10.4f} │ "
                    f"{metrics[f'recall_{class_name}']:>10.4f} │ "
                    f"{metrics[f'f1_{class_name}']:>10.4f} │ "
                    f"{support:>8.0f} │"
                )
        
        print(f"└{'─'*68}┘")
    
    # Confusion Matrix
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        num_classes = cm.shape[0]
        
        print(f"\nConfusion Matrix")
        print(f"┌{'─'*68}┐")
        
        if class_names is None:
            class_names = [f"C{i}" for i in range(num_classes)]
        
        header = "│ True\\Pred │ " + " │ ".join([f"{name[:8]:^8}" for name in class_names]) + " │"
        print(header)
        print(f"├{'─'*68}┤")
        
        for i, class_name in enumerate(class_names):
            row = f"│ {class_name[:10]:<10} │ "
            row += " │ ".join([f"{cm[i][j]:^8.0f}" for j in range(num_classes)])
            row += " │"
            print(row)
        
        print(f"└{'─'*68}┘")
    
    print("="*70 + "\n")


def save_results(metrics, save_path, class_names=None):
    """Save test results to file"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("MedMamba Test Results\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:           {metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Precision (Macro):  {metrics['precision_macro']:.4f}\n")
        f.write(f"  Recall (Macro):     {metrics['recall_macro']:.4f}\n")
        f.write(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}\n")
        
        if 'auc' in metrics:
            f.write(f"  AUC:                {metrics['auc']:.4f}\n")
        elif 'auc_macro' in metrics:
            f.write(f"  AUC (Macro):        {metrics['auc_macro']:.4f}\n")
            f.write(f"  AUC (Weighted):     {metrics['auc_weighted']:.4f}\n")
        
        f.write(f"\n")
        
        # Per-class metrics
        if class_names:
            f.write("Per-Class Metrics:\n")
            cm = metrics.get('confusion_matrix', None)
            for i, class_name in enumerate(class_names):
                if f'precision_{class_name}' in metrics:
                    support = cm[i].sum() if cm is not None else 0
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Precision: {metrics[f'precision_{class_name}']:.4f}\n")
                    f.write(f"    Recall:    {metrics[f'recall_{class_name}']:.4f}\n")
                    f.write(f"    F1-Score:  {metrics[f'f1_{class_name}']:.4f}\n")
                    f.write(f"    Support:   {support:.0f}\n")
            f.write("\n")
        
        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            f.write("Confusion Matrix:\n")
            cm = metrics['confusion_matrix']
            f.write(str(cm) + "\n")
    
    print(f"💾 Results saved to: {save_path}")


def main():
    # Load configuration
    config, is_3d = load_config(USE_CONFIG)
    
    print("="*70)
    print("MedMamba Testing")
    print("="*70)
    print(f"📝 Configuration: {USE_CONFIG}")
    print(f"🏗️  Model: {config.model.model_size}")
    print(f"📊 Classes: {config.model.num_classes} - {config.model.class_names}")
    print(f"💾 Checkpoint: {config.training.checkpoint_path}")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Print dataset paths being used
    print(f"\n📂 Dataset configuration:")
    if config.data.train_csv:
        print(f"   Train CSV: {config.data.train_csv}")
    if config.data.val_csv:
        print(f"   Val CSV: {config.data.val_csv}")
    if config.data.test_csv:
        print(f"   Test CSV: {config.data.test_csv}")
    print(f"   Test root: {config.data.test_root}")
    
    # Create test data loader
    train_loader, val_loader, test_loader, num_classes, class_to_idx = create_data_loaders(config)
    
    # Get class names from config or dataset
    class_names = getattr(config.model, 'class_names', None)
    if class_names is None and hasattr(test_loader.dataset, 'classes'):
        class_names = test_loader.dataset.classes
    
    if class_names:
        print(f"📋 Classes ({num_classes}): {', '.join(class_names)}")
    else:
        print(f"📋 Number of classes: {num_classes}")
    
    # Use model size from config
    model_size = config.model.model_size
    print(f"\n🏗️  Model configuration: {model_size.upper()}")
    
    # Create model
    model = create_model(
        model_name=model_size,
        config=config,
        num_classes=num_classes,
        is_3d=is_3d,
        patch_size=config.model.patch_size,
        in_chans=config.model.in_chans,
        d_state=config.model.d_state,
        drop_rate=config.model.drop_rate,
        attn_drop_rate=config.model.attn_drop_rate,
        drop_path_rate=config.model.drop_path_rate,
        use_checkpoint=False  # Disable gradient checkpointing for inference
    )
    
    # Load checkpoint
    model = load_model(config.training.checkpoint_path, model, device)
    model = model.to(device)
    
    # Evaluate model
    print("\n" + "="*70)
    print("Starting Evaluation...")
    print("="*70)
    
    metrics = evaluate_model(model, test_loader, device, num_classes, class_names)
    
    # Print results
    print_results(metrics, class_names)
    
    # Save results
    if SAVE_RESULTS_PATH:
        save_path = SAVE_RESULTS_PATH
    else:
        # Save to same directory as checkpoint
        checkpoint_dir = os.path.dirname(config.training.checkpoint_path)
        save_path = os.path.join(checkpoint_dir, 'test_results.txt')
    
    save_results(metrics, save_path, class_names)
    
    print("✅ Testing completed!")


if __name__ == '__main__':
    main()

