from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.policies.pi05.puma import PUMA_FUTURE_FEATURES, PUMA_FUTURE_VALID


class PUMASidecarDataset(Dataset):
    """Attach cached PUMA teacher features without changing the LeRobot dataset."""

    def __init__(self, dataset: Dataset, cache_root: str | Path):
        self.dataset = dataset
        self.cache_root = Path(cache_root)
        manifest = self.cache_root / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"PUMA cache manifest not found: {manifest}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    @lru_cache(maxsize=16)
    def _episode(self, episode_index: int) -> dict[str, np.ndarray]:
        path = self.cache_root / f"episode_{episode_index:06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing PUMA sidecar episode: {path}")
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        episode_index = int(torch.as_tensor(item["episode_index"]).item())
        frame_index = int(torch.as_tensor(item["index"]).item())
        episode = self._episode(episode_index)
        locations = np.flatnonzero(episode["frame_index"] == frame_index)
        if len(locations) != 1:
            raise KeyError(f"Frame {frame_index} is absent from PUMA episode {episode_index}")
        location = int(locations[0])
        item[PUMA_FUTURE_FEATURES] = torch.from_numpy(episode["features"][location]).float()
        item[PUMA_FUTURE_VALID] = torch.from_numpy(episode["valid"][location]).bool()
        return item
