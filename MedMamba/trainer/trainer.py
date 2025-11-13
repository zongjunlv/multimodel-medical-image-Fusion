from cProfile import label
import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

from utils import AverageMeter, MetricsCalculator, accuracy

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, dataloader):
        self.model.train()
        stats = {
            'total_loss' : 0.0,
            'processed_samples' : 0
        }
        
        pbar = tqdm(dataloader, desc='Train', bar_format='{l_bar}{bar:30}{r_bar}', colour='blue')

        for batch in pbar:
            img, label = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            logits = self.model(img)
            total_loss = self.criterion(logits, label)

            total_loss.backward()
            self.optimizer.step()

            batch_size = img.size(0)
            stats['total_loss'] += total_loss.item()
            stats['processed_samples'] += batch_size

            postfix = {
                'loss': f"{stats['total_loss']/stats['processed_samples']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            }

            pbar.set_postfix(postfix)

        return stats['total_loss'] / stats['processed_samples']
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        processed_sample = 0

        pbar = tqdm(dataloader, desc='val', bar_format='{l_bar}{bar:30}{r_bar}', colour='green')

        for batch in pbar:
            img, label = [x.to(self.device) for x in batch]

            logits = self.model(img)
            loss = self.criterion(logits, label)

            batch_size = img.size(0)
            total_loss += loss.item() * batch_size
            processed_sample += batch_size

            pbar.set_postfix({
                    'loss':f'{total_loss/processed_sample:.4f}'
                }
            )
        
        return total_loss / processed_sample





class MedMambaTrainer:
    """Trainer for MedMamba model"""
    
    def __init__(self, model, config, train_loader, val_loader, device, logger=None):
        # 存储训练核心对象与运行设备
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Metrics
        self.train_metrics = MetricsCalculator(config.model.num_classes)
        self.val_metrics = MetricsCalculator(config.model.num_classes)
        
        # Training state
        self.epoch = 0
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_precision = 0.0
        self.best_recall = 0.0
        self.best_sensitivity = 0.0
        self.best_specificity = 0.0
        self.best_auc = 0.0
        self.best_epoch = 0
        self.best_metrics = {}  # 保存所有最佳指标
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Early stopping
        self.early_stopping_counter = 0
        self.early_stopping_patience = getattr(config.training, 'early_stopping_patience', 20)
        self.use_early_stopping = getattr(config.training, 'use_early_stopping', True)
        
        # Save path
        self.save_path = os.path.join(config.training.save_dir, f'{config.model.model_name}_best.pth')
        
        if self.logger:
            self.logger.info(f"Trainer initialized with save path: {self.save_path}")
            if self.use_early_stopping:
                self.logger.info(f"Early stopping enabled with patience: {self.early_stopping_patience}")
    
    def _create_optimizer(self):
        """Create optimizer based on configuration"""
        # 根据配置关键字选择对应优化器
        if self.config.training.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        self.train_metrics.reset()  # 清空累计指标，避免跨 epoch 污染
        
        train_bar = tqdm(self.train_loader, file=sys.stdout, desc=f'Train Epoch [{self.epoch+1}/{self.config.training.epochs}]')
        
        end = time.time()
        for i, (images, labels) in enumerate(train_bar):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            images, labels = images.to(self.device), labels.to(self.device)  # 同步到目标设备
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  # 立即更新参数，未做梯度累积
            
            # Measure accuracy and record loss
            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            
            # Update metrics
            self.train_metrics.update(outputs, labels)  # 汇总混淆矩阵相关统计
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%'
            })
        
        return losses.avg, top1.avg
    
    def validate(self):
        """Validate the model and compute detailed metrics"""
        self.model.eval()
        
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        
        self.val_metrics.reset()
        
        # Suppress gradient checkpointing warning during validation
        # This warning appears because validation uses torch.no_grad() but model has use_checkpoint=True
        # It's safe to ignore as validation doesn't need gradients
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='None of the inputs have requires_grad=True')
            
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout, desc='Validation')
                
                end = time.time()
                for i, (images, labels) in enumerate(val_bar):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)  # 验证阶段只做前向和统计
                    
                    # Measure accuracy and record loss
                    acc1 = accuracy(outputs, labels, topk=(1,))[0]
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    
                    # Update metrics
                    self.val_metrics.update(outputs, labels)
                    
                    # Measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    
                    # Update progress bar
                    val_bar.set_postfix({
                        'loss': f'{losses.avg:.4f}',
                        'acc': f'{top1.avg:.2f}%'
                    })
        
        # Compute detailed metrics
        detailed_metrics = self.val_metrics.compute_metrics()
        
        return losses.avg, top1.avg, detailed_metrics
    
    def _log_detailed_metrics(self, metrics, prefix='Val'):
        """Log detailed metrics in table format"""
        if self.logger:
            # Overall metrics table
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"{prefix} Metrics Summary")
            self.logger.info(f"{'='*70}")
            
            # Overall metrics in table format
            self.logger.info(f"┌{'─'*68}┐")
            self.logger.info(f"│ {'Metric':<20} │ {'Value':<45} │")
            self.logger.info(f"├{'─'*68}┤")
            self.logger.info(f"│ {'Accuracy':<20} │ {metrics['accuracy']*100:>6.2f}% {' '*37} │")
            self.logger.info(f"│ {'Precision (Macro)':<20} │ {metrics['precision_macro']:>6.4f} {' '*38} │")
            self.logger.info(f"│ {'Sensitivity (Macro)':<20} │ {metrics['sensitivity_macro']:>6.4f} {' '*38} │")
            self.logger.info(f"│ {'Specificity (Macro)':<20} │ {metrics['specificity_macro']:>6.4f} {' '*38} │")
            self.logger.info(f"│ {'F1-Score (Macro)':<20} │ {metrics['f1_macro']:>6.4f} {' '*38} │")
            
            # Add AUC if available
            if 'auc' in metrics:
                self.logger.info(f"│ {'AUC':<20} │ {metrics['auc']:>6.4f} {' '*38} │")
            elif 'auc_macro' in metrics:
                self.logger.info(f"│ {'AUC (Macro)':<20} │ {metrics['auc_macro']:>6.4f} {' '*38} │")
                self.logger.info(f"│ {'AUC (Weighted)':<20} │ {metrics['auc_weighted']:>6.4f} {' '*38} │")
            
            self.logger.info(f"└{'─'*68}┘")
            
            # Per-class metrics table
            num_classes = self.config.model.num_classes
            if num_classes <= 10:  # Only show per-class for reasonable number of classes
                self.logger.info(f"\n{prefix} Per-Class Metrics")
                self.logger.info(f"┌{'─'*90}┐")
                self.logger.info(f"│ {'Class':<15} │ {'Precision':>10} │ {'Sensitivity':>11} │ {'Specificity':>11} │ {'F1-Score':>10} │ {'Support':>8} │")
                self.logger.info(f"├{'─'*90}┤")
                
                class_names = getattr(self.config.model, 'class_names', None)
                if class_names is None:
                    class_names = [f"Class_{i}" for i in range(num_classes)]
                
                for i, class_name in enumerate(class_names):
                    if f'precision_{class_name}' in metrics:
                        # Get support from confusion matrix
                        cm = metrics.get('confusion_matrix', None)
                        support = cm[i].sum() if cm is not None else 0
                        specificity = metrics.get(f'specificity_{class_name}', 0.0)
                        
                        self.logger.info(
                            f"│ {class_name:<15} │ "
                            f"{metrics[f'precision_{class_name}']:>10.4f} │ "
                            f"{metrics[f'recall_{class_name}']:>11.4f} │ "
                            f"{specificity:>11.4f} │ "
                            f"{metrics[f'f1_{class_name}']:>10.4f} │ "
                            f"{support:>8.0f} │"
                        )
                
                self.logger.info(f"└{'─'*90}┘")
            
            # Confusion Matrix
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                self.logger.info(f"\nConfusion Matrix")
                self.logger.info(f"┌{'─'*68}┐")
                
                # Header
                class_names = getattr(self.config.model, 'class_names', None)
                if class_names is None:
                    class_names = [f"C{i}" for i in range(num_classes)]
                
                header = "│ True\\Pred │ " + " │ ".join([f"{name[:8]:^8}" for name in class_names]) + " │"
                self.logger.info(header)
                self.logger.info(f"├{'─'*68}┤")
                
                # Matrix rows
                for i, class_name in enumerate(class_names):
                    row = f"│ {class_name[:10]:<10} │ "
                    row += " │ ".join([f"{cm[i][j]:^8.0f}" for j in range(num_classes)])
                    row += " │"
                    self.logger.info(row)
                
                self.logger.info(f"└{'─'*68}┘")
            
            self.logger.info(f"{'='*70}\n")
        else:
            # Print to console if no logger
            auc_str = ""
            if 'auc' in metrics:
                auc_str = f", AUC: {metrics['auc']:.3f}"
            elif 'auc_macro' in metrics:
                auc_str = f", AUC: {metrics['auc_macro']:.3f}"
            
            print(f"\n{prefix} Metrics: Acc={metrics['accuracy']*100:.2f}% "
                  f"P={metrics['precision_macro']:.3f} "
                  f"Sen={metrics['sensitivity_macro']:.3f} "
                  f"Spe={metrics['specificity_macro']:.3f} "
                  f"F1={metrics['f1_macro']:.3f}{auc_str}")
    
    def _print_best_metrics(self):
        """在训练结束时打印所有最佳指标"""
        if self.logger:
            self.logger.info('='*70)
            self.logger.info('🎉 训练完成！最佳模型性能总结')
            self.logger.info('='*70)
            self.logger.info(f'📍 最佳模型出现在 Epoch: {self.best_epoch}')
            self.logger.info('-'*70)
            self.logger.info('📊 总体指标:')
            self.logger.info(f'   • Accuracy:    {self.best_acc:.4f}% ({self.best_acc/100:.4f})')
            self.logger.info(f'   • Precision:   {self.best_precision:.4f}')
            self.logger.info(f'   • Sensitivity: {self.best_sensitivity:.4f}')
            self.logger.info(f'   • Specificity: {self.best_specificity:.4f}')
            self.logger.info(f'   • F1-Score:    {self.best_f1:.4f}')
            if self.best_auc > 0:
                self.logger.info(f'   • AUC:         {self.best_auc:.4f}')
            
            # 如果有每类别的指标，也打印出来
            if self.best_metrics:
                class_names = getattr(self.config.model, 'class_names', None)
                if class_names:
                    self.logger.info('-'*70)
                    self.logger.info('📋 各类别详细指标:')
                    self.logger.info(f"{'类别':<15} {'Precision':>10} {'Sensitivity':>11} {'Specificity':>11} {'F1-Score':>10} {'Support':>10}")
                    self.logger.info('-'*70)
                    
                    for i, class_name in enumerate(class_names):
                        precision_key = f'precision_{class_name}'
                        recall_key = f'recall_{class_name}'
                        specificity_key = f'specificity_{class_name}'
                        f1_key = f'f1_{class_name}'
                        
                        if precision_key in self.best_metrics:
                            # 从混淆矩阵获取support
                            cm = self.best_metrics.get('confusion_matrix', None)
                            support = cm[i].sum() if cm is not None else 0
                            specificity = self.best_metrics.get(specificity_key, 0.0)
                            
                            self.logger.info(
                                f"{class_name:<15} "
                                f"{self.best_metrics[precision_key]:>10.4f} "
                                f"{self.best_metrics[recall_key]:>11.4f} "
                                f"{specificity:>11.4f} "
                                f"{self.best_metrics[f1_key]:>10.4f} "
                                f"{support:>10.0f}"
                            )
                
                # 打印混淆矩阵
                if 'confusion_matrix' in self.best_metrics and class_names:
                    cm = self.best_metrics['confusion_matrix']
                    num_classes = len(class_names)
                    
                    self.logger.info('-'*70)
                    self.logger.info('📈 混淆矩阵:')
                    
                    # 动态调整列宽
                    max_name_len = max(len(name) for name in class_names)
                    col_width = max(8, max_name_len)
                    
                    # Header (使用变量避免f-string中的反斜杠)
                    label_text = "True\\Pred"
                    header = f"{label_text:<{col_width}} │ " + " │ ".join([f"{name[:col_width]:^{col_width}}" for name in class_names])
                    self.logger.info(header)
                    self.logger.info('-'*len(header))
                    
                    # Matrix rows
                    for i, class_name in enumerate(class_names):
                        row = f"{class_name[:col_width]:<{col_width}} │ "
                        row += " │ ".join([f"{cm[i][j]:^{col_width}.0f}" for j in range(num_classes)])
                        self.logger.info(row)
            
            self.logger.info('='*70)
        else:
            # Console output
            print('\n' + '='*70)
            print('🎉 训练完成！最佳模型性能总结')
            print('='*70)
            print(f'📍 最佳模型出现在 Epoch: {self.best_epoch}')
            print(f'📊 Accuracy: {self.best_acc:.2f}% | Precision: {self.best_precision:.4f} | '
                  f'Sensitivity: {self.best_sensitivity:.4f} | Specificity: {self.best_specificity:.4f} | '
                  f'F1: {self.best_f1:.4f}', end='')
            if self.best_auc > 0:
                print(f' | AUC: {self.best_auc:.4f}')
            else:
                print()
            print('='*70 + '\n')
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'best_precision': self.best_precision,
            'best_sensitivity': self.best_sensitivity,
            'best_specificity': self.best_specificity,
            'best_f1': self.best_f1,
            'best_auc': self.best_auc,
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, self.save_path)
            if self.logger:
                self.logger.info(f'✅ Best model saved to {self.save_path}')
        
        # Always save latest checkpoint
        latest_path = os.path.join(self.config.training.save_dir, f'{self.config.model.model_name}_latest.pth')
        torch.save(checkpoint, latest_path)
    
    def train(self):
        """Full training loop with early stopping"""
        if self.logger:
            self.logger.info(f"Starting training for {self.config.training.epochs} epochs")
        
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate and get detailed metrics
            val_loss, val_acc, val_metrics = self.validate()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc.item())
            
            # Extract key metrics
            val_f1 = val_metrics['f1_macro']
            val_precision = val_metrics['precision_macro']
            val_recall = val_metrics['recall_macro']
            val_sensitivity = val_metrics['sensitivity_macro']
            val_specificity = val_metrics['specificity_macro']
            
            # Extract AUC if available
            val_auc = None
            if 'auc' in val_metrics:
                val_auc = val_metrics['auc']
            elif 'auc_macro' in val_metrics:
                val_auc = val_metrics['auc_macro']
            
            # Log epoch results with detailed metrics
            if self.logger:
                auc_str = f', AUC: {val_auc:.3f}' if val_auc is not None else ''
                self.logger.info(
                    f'Epoch [{epoch+1}/{self.config.training.epochs}] '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                    f'P: {val_precision:.3f}, Sen: {val_sensitivity:.3f}, Spe: {val_specificity:.3f}, F1: {val_f1:.3f}{auc_str}'
                )
                
                # Log detailed metrics every N epochs
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    self._log_detailed_metrics(val_metrics, prefix='Validation')
            else:
                auc_str = f', AUC: {val_auc:.3f}' if val_auc is not None else ''
                print(
                    f'[Epoch {epoch+1}] Train Loss: {train_loss:.3f}, '
                    f'Val Acc: {val_acc:.3f}%, F1: {val_f1:.3f}{auc_str}'
                )
            
            # Check if this is the best model (based on AUC for medical imaging)
            current_auc = val_auc if val_auc is not None else 0.0
            is_best = current_auc > self.best_auc
            if is_best:
                self.best_auc = current_auc
                self.best_acc = val_acc.item()
                self.best_precision = val_precision
                self.best_recall = val_recall
                self.best_sensitivity = val_sensitivity
                self.best_specificity = val_specificity
                self.best_f1 = val_f1
                self.best_epoch = epoch + 1
                self.best_metrics = val_metrics.copy()  # 保存所有验证指标
                self.early_stopping_counter = 0  # Reset early stopping counter
                if self.logger:
                    self.logger.info(f'🎉 New best AUC: {self.best_auc:.4f}')
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            if self.config.training.save_best_only:
                if is_best:
                    self.save_checkpoint(is_best=True)
            else:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping check
            if self.use_early_stopping:
                if self.early_stopping_counter >= self.early_stopping_patience:
                    if self.logger:
                        self.logger.info(
                            f'⏹️  Early stopping triggered! '
                            f'No improvement for {self.early_stopping_patience} epochs.'
                        )
                    else:
                        print(f'Early stopping at epoch {epoch+1}')
                    break
                
                # Log early stopping status
                if self.early_stopping_counter > 0 and self.logger:
                    self.logger.info(
                        f'⚠️  Early stopping counter: {self.early_stopping_counter}/'
                        f'{self.early_stopping_patience}'
                    )
        
        # Print detailed best metrics
        self._print_best_metrics()
        
        return {
            'best_acc': self.best_acc,
            'best_precision': self.best_precision,
            'best_sensitivity': self.best_sensitivity,
            'best_specificity': self.best_specificity,
            'best_f1': self.best_f1,
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
