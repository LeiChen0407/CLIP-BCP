import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from models import ClipBcpFusionModel
from data import DeepfakeDataset
from training import train
from utils import visualize_training_history, print_run_arguments, CLIP_MEAN, CLIP_STD


def main():
    parser = argparse.ArgumentParser(description='Train the CLIP-BCP fusion Model')
    
    # Data Parameters
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Training Set Directory Path')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Verification Set Directory Path')
    parser.add_argument('--max_samples', type=int, default=-1)
    
    # Model Parameters
    parser.add_argument('--clip_model_path', type=str, required=True,
                        help='CLIP Pre-trained Model Path')
    
    # Default parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    

    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output Directory')
    parser.add_argument('--seed', type=int, default=44)
    
    args = parser.parse_args()
    
    print_run_arguments(args)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])
    
    max_s = None if args.max_samples == -1 else args.max_samples
    train_dataset = DeepfakeDataset(args.train_dir, transform, max_s)
    val_dataset = DeepfakeDataset(args.val_dir, transform)
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    model = ClipBcpFusionModel(args.clip_model_path, normalize_features=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    model_save_path = os.path.join(args.output_dir, 'best_model.pth')
    print("\nStarting training...")
    
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                   device, args.num_epochs, model_save_path, args.output_dir)
    
    visualize_training_history(history, os.path.join(args.output_dir, 'training_history.png'))
    
    checkpoint = torch.load(model_save_path, map_location=device)
    print(f"\nBest Val Acc: {checkpoint.get('val_acc', 0):.4f}")


if __name__ == '__main__':
    main()

