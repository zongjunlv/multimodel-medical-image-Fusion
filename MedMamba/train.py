import os
from matplotlib.style import available
import time
import random
import numpy as np
import swanlab
import random
from datetime import timedelta
from timm.models.layers import weight_init
import torch
from torch.cuda import Device
from torch.optim import optimizer
from torch.utils.data import DataLoader
import torch.nn as nn


from trainer import Trainer
from utils import compute_all_metrics
from data.medical_dataset import Medical_Dataset
from models import Model
from utils.evaluator import evaluate_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():

    print("\n" + "="*60)
    print(f"{'Model Training Pipeline':^60}")
    print("="*60)

    start_time = time.time()
    seed = 3407
    set_seed(seed)
    
    

    train_dataset = Medical_Dataset(mode='train')
    val_dataset = Medical_Dataset(mode='val')

    batch_size = 8
    lr = 1e-5
    weight_decay = 1e-2

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, drop_last=True,
                                        num_workers=12,persistent_workers=True, prefetch_factor=8)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True,
                                        num_workers=12,persistent_workers=True, prefetch_factor=8)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model = Model()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay )
    criterion = nn.CrossEntropyLoss()

    print("\nCONFIGURATION")
    print(f"  {'Batch size':18}: {batch_size}")
    print(f"  {'Learning rate':18}: {lr}")
    print(f"  {'Device':18}: {device}")
    print(f"  {'Mixed Precision':18}: {'Enabled' if device.type == 'cuda' else 'Disabled (CPU)'}")

    print("\nPREPARING DATA...")


    trainer = Trainer(model, optimizer, criterion, device)                                      

    num_epochs = 100
    best_val_loss = float('inf')
    early_stop = 0
    patience = 10

    print("\n" + "-"*60)
    print(f"{'TRAINING STARTED':^60}")
    print("-"*60)
    
    swanlab.init(
    # 设置项目名
        project="my-awesome-project",
        
        # 设置超参数
        config={
            "learning_rate": lr,
            "architecture": "ABUS",
            "epochs": num_epochs,
            "logdir": "/data02/workspace/LZJ_SPACE/MedMamba/assets/logs"
        }
    )
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = trainer.train(train_dataloader)
        val_loss = trainer.validate(val_dataloader)

        epoch_time = time.time() - epoch_start
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}   Time: {str(timedelta(seconds=int(epoch_time)))}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = 'assets/checkpoints_3d'
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  Weights updated!")
            early_stop = 0
        else:
            early_stop += 1
            print(f"  Early stopping count: {early_stop}/{patience}")
            if early_stop == patience:
                print(f"  Early stopping triggered at epoch {epoch + 1}.")
                print(f"  Best model saved as: best_model.pth")
                break
        
        if epoch % 5 == 0:
            accuracy, auc, sensitivity, specificity, f1, mcc = evaluate_model(model, val_dataloader, device, verbose=True)
        else:
            accuracy, auc, sensitivity, specificity, f1, mcc = evaluate_model(model, val_dataloader, device, verbose=False)
        swanlab.log({"train_loss":train_loss,
                    "val_loss": val_loss, 
                    "acc":accuracy,
                    "auc":auc, 
                    "sensitivity":sensitivity, 
                    "specificity":specificity, 
                    "f1":f1, 
                    "mcc":mcc} 
        )
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Training completed in {str(timedelta(seconds=int(total_time)))}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60 + "\n")
    swanlab.finish()
                


if __name__ == '__main__':
    main()
