"""Generate immutable sidecar teacher features for PUMA training.

This command only reads a LeRobot dataset. It refuses to place its output inside
the dataset root, so the source Parquet files and videos cannot be modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


class PUMATeacher:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from transformers import (
                AutoImageProcessor,
                AutoModel,
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Sam2Model,
                Sam2Processor,
            )
        except ImportError as exc:
            raise ImportError(
                "Install a Transformers release with Grounding DINO and SAM2 support"
            ) from exc

        self.device = torch.device(args.device)
        self.dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        self.ground_processor = AutoProcessor.from_pretrained(args.grounding_model)
        self.ground_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.grounding_model, dtype=self.dtype
        ).to(self.device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(args.sam_model)
        self.sam_model = Sam2Model.from_pretrained(args.sam_model, dtype=self.dtype).to(self.device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)
        self.dino_model = AutoModel.from_pretrained(args.dino_model, dtype=self.dtype).to(self.device).eval()
        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.feature_dim = int(self.dino_model.config.hidden_size)

    def _prepare_inputs(self, inputs: dict[str, object]) -> dict[str, object]:
        prepared = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                prepared[key] = value
            elif value.is_floating_point():
                prepared[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                prepared[key] = value.to(self.device)
        return prepared

    @torch.inference_mode()
    def __call__(self, image: Image.Image, prompt: str) -> tuple[np.ndarray, bool, float]:
        ground = self.ground_processor(images=image, text=prompt, return_tensors="pt")
        ground = self._prepare_inputs(ground)
        outputs = self.ground_model(**ground)
        detections = self.ground_processor.post_process_grounded_object_detection(
            outputs,
            ground["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        if len(detections["boxes"]) == 0:
            return np.zeros(self.dino_model.config.hidden_size, dtype=np.float32), False, 0.0
        best = int(torch.argmax(detections["scores"]).item())
        box = detections["boxes"][best].detach().cpu().tolist()
        score = float(detections["scores"][best].item())

        sam = self.sam_processor(images=image, input_boxes=[[[box]]], return_tensors="pt")
        sam = self._prepare_inputs(sam)
        sam_outputs = self.sam_model(**sam, multimask_output=False)
        masks = self.sam_processor.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(), sam["original_sizes"].detach().cpu()
        )
        mask = masks[0][0, 0].float()

        dino = self.dino_processor(images=image, return_tensors="pt")
        dino = self._prepare_inputs(dino)
        tokens = self.dino_model(**dino).last_hidden_state[:, 1:].float()
        side = int(math.sqrt(tokens.shape[1]))
        if side * side != tokens.shape[1]:
            raise RuntimeError(f"DINO patch count {tokens.shape[1]} is not square")
        mask = F.interpolate(mask[None, None], size=(side, side), mode="nearest").flatten()
        selected = mask > 0.5
        if not selected.any():
            return np.zeros(tokens.shape[-1], dtype=np.float32), False, score
        feature = tokens[0, selected].mean(dim=0)
        return feature.cpu().numpy().astype(np.float32), True, score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--target-map", type=Path, required=True, help="JSON mapping task text/index to target prompt")
    parser.add_argument("--future-steps", type=int, default=4)
    parser.add_argument("--future-stride", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    return parser.parse_args()


def to_pil(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().cpu().numpy()
    if array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array[..., :3])


def episode_cache_is_complete(path: Path, expected_length: int, future_steps: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            return (
                set(data.files) >= {"frame_index", "features", "valid", "scores"}
                and data["frame_index"].shape == (expected_length,)
                and data["features"].shape[:2] == (expected_length, future_steps)
                and data["valid"].shape == (expected_length, future_steps)
                and data["scores"].shape == (expected_length, future_steps)
            )
    except (OSError, ValueError, KeyError):
        return False


def save_episode(
    destination: Path,
    rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            frame_index=np.asarray([row[0] for row in rows], dtype=np.int64),
            features=np.stack([row[1] for row in rows]),
            valid=np.stack([row[2] for row in rows]),
            scores=np.stack([row[3] for row in rows]),
        )
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise ValueError("--output-dir must be outside --dataset-root; source datasets are read-only")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_map = json.loads(args.target_map.read_text())

    manifest = {
        "repo_id": args.repo_id,
        "dataset_root": str(dataset_root),
        "camera": args.camera,
        "future_steps": args.future_steps,
        "future_stride": args.future_stride,
        "models": {
            "grounding": args.grounding_model,
            "sam": args.sam_model,
            "dino": args.dino_model,
        },
        "target_map": target_map,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest != manifest:
            raise ValueError(
                f"Existing PUMA cache uses different settings: {manifest_path}. "
                "Choose a new output directory or remove the old cache."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    offsets = [0] + [(i + 1) * args.future_stride for i in range(args.future_steps)]
    metadata = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    pending_episodes = []
    for episode_index in range(len(metadata.episodes)):
        episode = metadata.episodes[episode_index]
        expected_length = int(episode["dataset_to_index"] - episode["dataset_from_index"])
        destination = output_dir / f"episode_{int(episode_index):06d}.npz"
        if episode_cache_is_complete(destination, expected_length, args.future_steps):
            print(f"Skipping completed episode {episode_index}", flush=True)
        else:
            pending_episodes.append(int(episode_index))

    if not pending_episodes:
        print("PUMA feature cache is already complete.", flush=True)
        return

    dataset = LeRobotDataset(
        args.repo_id,
        root=dataset_root,
        delta_timestamps={args.camera: [offset / metadata.fps for offset in offsets]},
        return_uint8=True,
    )
    teacher = PUMATeacher(args)
    for episode_index in pending_episodes:
        episode = metadata.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        rows = []
        teacher_cache: dict[tuple[int, str], tuple[np.ndarray, bool, float]] = {}
        print(
            f"Processing episode {episode_index} ({episode_end - episode_start} frames)", flush=True
        )
        for item_index in range(episode_start, episode_end):
            item = dataset[item_index]
            task = str(item.get("task", ""))
            task_index = str(int(torch.as_tensor(item.get("task_index", -1)).item()))
            prompt = target_map.get(task, target_map.get(task_index))
            if prompt is None:
                raise KeyError(f"No target prompt for task={task!r}, task_index={task_index}")
            frames = item[args.camera]
            frame_index = int(torch.as_tensor(item["index"]).item())
            features, valid, scores = [], [], []
            padding = item[f"{args.camera}_is_pad"][1:]
            for future_idx, frame in enumerate(frames[1:]):
                if bool(padding[future_idx]):
                    feature = np.zeros(teacher.feature_dim, dtype=np.float32)
                    is_valid, score = False, 0.0
                else:
                    future_frame_index = frame_index + offsets[future_idx + 1]
                    cache_key = (future_frame_index, prompt)
                    if cache_key not in teacher_cache:
                        teacher_cache[cache_key] = teacher(to_pil(frame), prompt)
                    feature, is_valid, score = teacher_cache[cache_key]
                features.append(feature)
                valid.append(is_valid and not bool(padding[future_idx]))
                scores.append(score)
            rows.append(
                (frame_index, np.stack(features), np.asarray(valid), np.asarray(scores, dtype=np.float32))
            )

        destination = output_dir / f"episode_{episode_index:06d}.npz"
        save_episode(destination, rows)
        print(f"Saved episode {episode_index} to {destination}", flush=True)


if __name__ == "__main__":
    main()
