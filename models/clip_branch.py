"""CLIP Branch"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class CLIPEmbeddingBranch(nn.Module):

    def __init__(
        self,
        model_path: str,
        clip_layer: int = 13,
        freeze_backbone: bool = True,
    ):

        super().__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained(model_path)
        self.clip_layer = clip_layer
        self.output_dim = self.clip_vision.config.hidden_size

        if freeze_backbone:
            for param in self.clip_vision.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.clip_vision(x, output_attentions=False, output_hidden_states=True)
        return outputs.hidden_states[self.clip_layer + 1][:, 0]
