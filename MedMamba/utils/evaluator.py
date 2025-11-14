import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from .metrics import compute_all_metrics


def evaluate(model, dataloader, device):

    model.to(device).eval()
    labels_list, probs_list, preds_list = [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Test', 
                       bar_format='{l_bar}{bar:30}{r_bar}',
                       colour='green')
        
        for img, labels in pbar:
            img, labels = img.to(device), labels.to(device)

            # 前向传播
            model_output = model(img)
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output
            
            # 计算概率和预测
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # 收集结果
            labels_list.append(labels.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds.cpu().numpy())

    # 合并所有批次的结果
    labels_arr = np.concatenate(labels_list, axis=0)
    probs_arr = np.concatenate(probs_list, axis=0)
    preds_arr = np.concatenate(preds_list, axis=0)
    
    # 计算指标
    all_metrics = compute_all_metrics(labels_arr, preds_arr, probs_arr)
    
    accuracy = all_metrics['accuracy']
    auc = all_metrics.get('auc', 0.0)
    sensitivity = all_metrics['sensitivity'] 
    specificity = all_metrics['specificity']
    f1 = all_metrics['macro_f1']
    mcc = all_metrics['mcc']
    
    return accuracy, auc, sensitivity, specificity, f1, mcc


def evaluate_model(model, dataloader, device, verbose=True):

    accuracy, auc, sensitivity, specificity, f1, mcc = evaluate(model, dataloader, device)
    
    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC:      {auc:.4f}")
        print(f"Sens:     {sensitivity:.4f}")
        print(f"Spec:     {specificity:.4f}")
        print(f"F1:       {f1:.4f}")
        print(f"MCC:      {mcc:.4f}")
    
    return accuracy, auc, sensitivity, specificity, f1, mcc


