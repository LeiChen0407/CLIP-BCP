"""Evaluator"""
import os
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, average_precision_score


def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    domain_results = {}
    
    with torch.no_grad():
        for images, labels, img_paths in tqdm(data_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            for i, img_path in enumerate(img_paths):
                domain = img_path.split(os.sep)[-3]
                if domain not in domain_results:
                    domain_results[domain] = {'preds': [], 'labels': [], 'probs': []}
                domain_results[domain]['preds'].append(preds[i].item())
                domain_results[domain]['labels'].append(labels[i].item())
                domain_results[domain]['probs'].append(probs[i, 1].item())
    
    overall_metrics = {}
    domain_metrics = {}
    rounded_domain_accs = []
    rounded_domain_aps = []
    for domain, results in domain_results.items():
        if len(results['labels']) > 1 and len(set(results['labels'])) > 1:
            d_acc = accuracy_score(results['labels'], results['preds'])
            d_ap = average_precision_score(results['labels'], results['probs'])
            domain_metrics[domain] = {'accuracy': d_acc, 'ap': d_ap}
            rounded_domain_accs.append(round(d_acc, 4))
            rounded_domain_aps.append(round(d_ap, 4))
    if rounded_domain_accs:
        mean_domain_acc = sum(rounded_domain_accs) / len(rounded_domain_accs)
    else:
        mean_domain_acc = accuracy_score(all_labels, all_preds)
    overall_metrics['accuracy'] = mean_domain_acc
    if rounded_domain_aps:
        overall_metrics['ap'] = sum(rounded_domain_aps) / len(rounded_domain_aps)
    else:
        overall_metrics['ap'] = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return overall_metrics, domain_metrics


def test_by_domain(model, data_loader, device):
    model.eval()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    domain_results = {}
    
    with torch.no_grad():
        for images, labels, img_paths in tqdm(data_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            for i, img_path in enumerate(img_paths):
                domain = img_path.split(os.sep)[-3]
                if domain not in domain_results:
                    domain_results[domain] = {'preds': [], 'labels': [], 'probs': []}
                domain_results[domain]['preds'].append(preds[i].item())
                domain_results[domain]['labels'].append(labels[i].item())
                domain_results[domain]['probs'].append(probs[i, 1].item())
    domain_names = sorted(domain_results.keys())
    all_acc = []
    all_ap = []
    
    for domain_idx, domain in enumerate(domain_names):
        results = domain_results[domain]
        if len(results['labels']) > 1 and len(set(results['labels'])) > 1:
            acc = accuracy_score(results['labels'], results['preds'])
            ap = average_precision_score(results['labels'], results['probs'])
            rounded_acc = round(acc, 4)
            rounded_ap = round(ap, 4)
            all_acc.append(rounded_acc)
            all_ap.append(rounded_ap)
            print(f"({domain_idx} {domain}): Acc: {rounded_acc:.4f}, AP: {rounded_ap:.4f}")
        else:
            acc = accuracy_score(results['labels'], results['preds'])
            rounded_acc = round(acc, 4)
            all_acc.append(rounded_acc)
            print(f"({domain_idx} {domain}): Acc: {rounded_acc:.4f}, AP: N/A")
    
    mean_acc = np.mean(all_acc) if all_acc else 0.0
    if len(all_ap) > 0:
        mean_ap = np.mean(all_ap)
        print(f"({len(domain_names)} Mean): Acc: {mean_acc:.4f}, AP: {mean_ap:.4f}")
    else:
        print(f"({len(domain_names)} Mean): Acc: {mean_acc:.4f}, AP: N/A")
    
    print("***************************")

