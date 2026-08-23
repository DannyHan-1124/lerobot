import json

import numpy as np
import torch

from lerobot.datasets.puma_sidecar import PUMASidecarDataset
from lerobot.policies.pi05.configuration_puma import PUMAConfig
from lerobot.policies.pi05.puma import PUMA_FUTURE_FEATURES, PUMA_FUTURE_VALID, dense_flow_rgb


def test_puma_config_validates_horizons():
    config = PUMAConfig(enabled=True, history_steps=4, history_stride=4)
    assert config.future_steps == 4


def test_dense_flow_rgb_shape_and_range():
    historical = torch.zeros(1, 2, 3, 32, 32)
    current = torch.zeros(1, 3, 32, 32)
    historical[:, :, :, 8:20, 5:17] = 1
    current[:, :, 8:20, 10:22] = 1
    flow = dense_flow_rgb(historical, current, resolution=(32, 32), noise_threshold=0.0)
    assert flow.shape == (1, 2, 3, 32, 32)
    assert 0 <= flow.min() <= flow.max() <= 1
    assert flow.max() > 0


class _Dataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"episode_index": torch.tensor(2), "index": torch.tensor(17)}


def test_sidecar_adds_features_without_modifying_source(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 1}))
    np.savez_compressed(
        tmp_path / "episode_000002.npz",
        frame_index=np.asarray([17]),
        features=np.ones((1, 4, 8), dtype=np.float32),
        valid=np.ones((1, 4), dtype=bool),
    )
    dataset = PUMASidecarDataset(_Dataset(), tmp_path)
    item = dataset[0]
    assert item[PUMA_FUTURE_FEATURES].shape == (4, 8)
    assert item[PUMA_FUTURE_VALID].all()
