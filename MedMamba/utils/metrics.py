import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MetricsCalculator:
    """Calculate various metrics for medical image classification"""
    
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and targets"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, outputs, targets):
        """
        Update with batch predictions
        
        Args:
            outputs: Model outputs (logits)
            targets: Ground truth labels
        """
        # Convert to probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Get predictions
        _, predicted = torch.max(outputs, 1)
        
        # Move to CPU and convert to numpy
        predicted = predicted.cpu().numpy()
        targets = targets.cpu().numpy()
        probs = probs.detach().cpu().numpy()
        
        # Store
        self.all_predictions.extend(predicted)
        self.all_targets.extend(targets)
        self.all_probabilities.extend(probs)
    
    def compute_metrics(self):
        """
        Compute comprehensive metrics
        
        Returns:
            dict: Dictionary of computed metrics
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        y_prob = np.array(self.all_probabilities)
        
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class and macro/weighted metrics
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Sensitivity (same as recall/sensitivity)
        metrics['sensitivity_macro'] = metrics['recall_macro']
        metrics['sensitivity_weighted'] = metrics['recall_weighted']
        
        # Specificity
        cm = confusion_matrix(y_true, y_pred)
        specificity_per_class = []
        
        for i in range(self.num_classes):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
            metrics[f'specificity_{self.class_names[i]}'] = specificity
        
        metrics['specificity_macro'] = np.mean(specificity_per_class)
        metrics['specificity_weighted'] = np.average(specificity_per_class, weights=[np.sum(cm[i, :]) for i in range(self.num_classes)])
        
        # AUC (for multiclass)
        try:
            if self.num_classes == 2:
                # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multiclass classification
                metrics['auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except ValueError:
            # Handle cases where AUC cannot be computed
            pass
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def get_classification_report(self):
        """Get detailed classification report"""
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        )
    
    def compute_specificity(self):
        """Compute specificity per class"""
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_predictions)
        
        cm = confusion_matrix(y_true, y_pred)
        specificity_per_class = []
        
        for i in range(self.num_classes):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(specificity)
        
        return {
            'specificity_macro': np.mean(specificity_per_class),
            'specificity_per_class': specificity_per_class
        }


def print_metrics_summary(metrics, class_names=None):
    """Print a formatted summary of metrics"""
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    
    # Overall metrics
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    
    if 'auc' in metrics:
        print(f"AUC: {metrics['auc']:.4f}")
    elif 'auc_macro' in metrics:
        print(f"AUC (Macro): {metrics['auc_macro']:.4f}")
        print(f"AUC (Weighted): {metrics['auc_weighted']:.4f}")
    
    print(f"\nPrecision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    
    print(f"\nPrecision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print("="*80)
