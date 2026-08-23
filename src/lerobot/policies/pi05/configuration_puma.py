from dataclasses import dataclass


@dataclass
class PUMAConfig:
    """Dynamics-aware training and inference settings for PI0.5."""

    enabled: bool = False
    history_steps: int = 4
    history_stride: int = 4
    flow_resolution: tuple[int, int] = (64, 64)
    flow_camera_key: str = "observation.images.base_0_rgb"
    dataset_flow_camera_key: str | None = None
    flow_magnitude_percentile: float = 99.0
    flow_noise_threshold: float = 0.05
    future_steps: int = 4
    future_stride: int = 4
    future_feature_dim: int = 768
    world_loss_weight: float = 0.05
    feature_cache: str | None = None

    def __post_init__(self) -> None:
        if self.history_steps < 1 or self.history_stride < 1:
            raise ValueError("PUMA history_steps and history_stride must be positive")
        if self.future_steps < 1 or self.future_stride < 1:
            raise ValueError("PUMA future_steps and future_stride must be positive")
        if self.future_feature_dim < 1:
            raise ValueError("PUMA future_feature_dim must be positive")
        if self.world_loss_weight < 0:
            raise ValueError("PUMA world_loss_weight cannot be negative")
