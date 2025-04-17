from pathlib import Path
from typing import Callable
import torch
import numpy as np

from .lerobot_dataset.lerobot_dataset import LeRobotDataset


class RJLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        data = {
            "image": {
                "observation.images.0_top": self.tensor2image(
                    item["observation.images.0_top"]
                ),
                "observation.images.1_right": self.tensor2image(
                    item["observation.images.1_right"]
                ),
            },
            "text": {"task": item["task"]},
            "point_cloud": {},
            "struct_data": {"observation.state": item["observation.state"]},
        }
        return data

    def tensor2image(self, tensor, imtype=np.uint8):
        """ "Converts a Tensor array into a numpy image array."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        data = np.clip(tensor, 0, 1)
        data_uint8 = (data * 255).astype(np.uint8)
        return data_uint8
