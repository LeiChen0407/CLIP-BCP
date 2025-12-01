"""Fusion"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_branch import CLIPEmbeddingBranch
from .bcp_branch import BrightChannelPriorBranch


class ClipBcpFusionModel(nn.Module):

    def __init__(self, model_path: str, normalize_features: bool = False):
        super().__init__()
        self.normalize_features = normalize_features

        self.clip_branch = CLIPEmbeddingBranch(model_path)
        self.bcp_branch = BrightChannelPriorBranch()

        fusion_dim = self.clip_branch.output_dim + self.bcp_branch.output_dim
        self.fc1 = nn.Linear(fusion_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clip_vec = self.clip_branch(x)
        bcp_vec = self.bcp_branch(x)

        if self.normalize_features:
            clip_vec = F.normalize(clip_vec, p=2, dim=1)
            bcp_vec = F.normalize(bcp_vec, p=2, dim=1)

        fused = torch.cat([clip_vec, bcp_vec], dim=1)
        h1 = F.gelu(self.fc1(fused))
        h2 = F.gelu(self.fc2(h1))
        return self.fc3(h2)
