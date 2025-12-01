import os
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import ClipBcpFusionModel
from data import DeepfakeDataset
from training import test_by_domain
from utils import CLIP_MEAN, CLIP_STD


def main():
    parser = argparse.ArgumentParser(description='Testing the CLIP-BCP fusion Model')
    
    # Required Parameters
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Test dataset directory path')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='checkpoint file path (.pth)')
    parser.add_argument('--clip_model_path', type=str, required=True,
                        help='CLIP Pre-trained Model Path')
    
    # Default parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    if not args.checkpoint_path.endswith('.pth') or not os.path.exists(args.checkpoint_path):
        print(f"Error: Invalid model path: {args.checkpoint_path}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test dir not found: {args.test_dir}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])
    
    test_dataset = DeepfakeDataset(args.test_dir, transform)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Test: {len(test_dataset)}")
    
    model = ClipBcpFusionModel(args.clip_model_path, normalize_features=True).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded: {args.checkpoint_path}")
    
    print("\n" + "=" * 50)
    test_by_domain(model, test_loader, device)
    print("=" * 50)


if __name__ == '__main__':
    main()

