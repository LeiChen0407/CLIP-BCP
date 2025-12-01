"""BCP Branch"""
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import kornia.morphology as KM


class BrightChannelPriorBranch(nn.Module):

    def __init__(
        self,
        ms_kernels: Sequence[int] | None = None,
        fc_dim: int = 1024,
    ):
        super().__init__()
        if ms_kernels is None:
            ms_kernels = (1, 3)
        self.ms_kernels: Iterable[int] = tuple(ms_kernels)
        in_channels = len(self.ms_kernels)

        self.mapper = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, fc_dim)
        self.output_dim = fc_dim

    @staticmethod
    def _denormalize_clip_to_255(x: torch.Tensor) -> torch.Tensor:
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x_01 = (x * clip_std + clip_mean).clamp(0.0, 1.0)
        return (x_01 * 255.0).clamp(0.0, 255.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_255 = self._denormalize_clip_to_255(x)
        bright = x_255.max(dim=1, keepdim=True).values
        dilated_list = [
            KM.dilation(bright, torch.ones(k, k, device=x.device, dtype=x.dtype))
            for k in self.ms_kernels
        ]
        bright_ms = torch.cat(dilated_list, dim=1)
        feat = self.mapper(bright_ms).flatten(1)
        return self.fc(feat)
