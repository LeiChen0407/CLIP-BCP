"""Trainer"""
import os
import json
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from .evaluator import evaluate
from utils.metrics import round_metrics


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, 
          device, num_epochs, save_path='best_model.pth', output_dir=None):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, all_preds, all_labels = 0.0, [], []
        
        for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            all_preds.extend(torch.max(logits, 1)[1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)

        overall_metrics, domain_metrics = evaluate(model, val_loader, device)
        val_acc = overall_metrics['accuracy']
        val_loss = 1.0 - val_acc
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)
            print(f"Model saved (Val Acc: {val_acc:.4f})")

            if output_dir:
                metrics_data = {
                    'overall': round_metrics(overall_metrics),
                    'domains': round_metrics(domain_metrics)
                }
                with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
                    json.dump(metrics_data, f, indent=2)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
    
    return history

