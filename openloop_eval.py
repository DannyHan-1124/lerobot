import json
import os
import random
from types import SimpleNamespace

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def namespace_to_dict(value):
    if isinstance(value, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [namespace_to_dict(v) for v in value]
    return value


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch


def ensure_attr(obj, name, value):
    if not hasattr(obj, name):
        setattr(obj, name, value)


def get_episode_index(sample):
    if "episode_index" in sample:
        ep = sample["episode_index"]
    elif "episode.idx" in sample:
        ep = sample["episode.idx"]
    else:
        raise KeyError(f"No episode index key found. Available keys: {sample.keys()}")

    if isinstance(ep, torch.Tensor):
        ep = ep.item()

    return int(ep)


def predict_one_sample(policy, preprocessor, postprocessor, sample, device):
    processed_batch = preprocessor(sample)
    processed_batch = move_to_device(processed_batch, device)

    with torch.no_grad():
        pred_action = policy.predict_action_chunk(processed_batch)[:, 0]

    pred_action = postprocessor(pred_action)

    pred = pred_action.squeeze(0).detach().to(torch.float32).cpu()
    gt = sample["action"].detach().to(torch.float32).cpu()

    if pred.ndim == 2:
        pred = pred[0]
    if gt.ndim == 2:
        gt = gt[0]

    return gt, pred

# =========================
# Config
# =========================

ckpt_dir = "/hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_bs256_10ksteps/checkpoints/last/pretrained_model"

config_path = "/hkfs/work/workspace/scratch/utphd-myspace/outputs/pi05_bs256_10ksteps/checkpoints/last/pretrained_model/train_config.json"

plot_dir = "/hkfs/work/workspace/scratch/utphd-myspace/lerobot/openloop_eval/pi05_bs256_10ksteps"
os.makedirs(plot_dir, exist_ok=True)

num_episodes_to_plot = 3
random_seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"

print("================================")
print("Open-loop evaluation")
print("================================")
print("checkpoint:", ckpt_dir)
print("config:", config_path)
print("device:", device)

with open(config_path, "r") as f:
    cfg = dict_to_namespace(json.load(f))

# Defaults needed by the current make_dataset(cfg).
ensure_attr(cfg.dataset, "image_transforms", SimpleNamespace(enable=False))
ensure_attr(cfg.dataset, "revision", None)
ensure_attr(cfg.dataset, "root", None)
ensure_attr(cfg.dataset, "episodes", None)
ensure_attr(cfg.dataset, "streaming", False)
ensure_attr(cfg.dataset, "video_backend", "torchcodec")
ensure_attr(cfg.dataset, "use_imagenet_stats", True)
ensure_attr(cfg, "num_workers", 4)
ensure_attr(cfg, "tolerance_s", 1e-4)
ensure_attr(cfg, "rename_map", {})
cfg.rename_map = namespace_to_dict(cfg.rename_map)
ensure_attr(cfg.policy, "observation_delta_indices", None)
ensure_attr(cfg.policy, "action_delta_indices", None)
ensure_attr(cfg.policy, "reward_delta_indices", None)
ensure_attr(cfg, "trainable_config", cfg.policy)

dataset = make_dataset(cfg)

print("\n================================")
print("Dataset info")
print("================================")
print("dataset length:", len(dataset))

sample0 = dataset[0]
print("sample keys:", sample0.keys())
print("action shape:", sample0["action"].shape)

if hasattr(dataset.meta, "fps"):
    print("dataset fps:", dataset.meta.fps)


print("\n================================")
print("Loading policy")
print("================================")

policy = PI05Policy.from_pretrained(ckpt_dir)
policy.config.device = device
policy.config.compile_model = False
policy.to(device)
policy.eval()

print("Loaded policy from:", ckpt_dir)

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=ckpt_dir,
    preprocessor_overrides={
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    },
    postprocessor_overrides={
        "unnormalizer_processor": {
            "stats": dataset.meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    },
)


# =========================
# Collect episode indices
# =========================

episode_to_indices = {}

for idx in range(len(dataset)):
    sample = dataset[idx]
    ep = get_episode_index(sample)
    episode_to_indices.setdefault(ep, []).append(idx)

all_episodes = sorted(episode_to_indices.keys())

print("\n================================")
print("Episode info")
print("================================")
print("num episodes:", len(all_episodes))
print("first episodes:", all_episodes[:10])

random.seed(random_seed)

if len(all_episodes) <= num_episodes_to_plot:
    target_episodes = all_episodes
else:
    target_episodes = random.sample(all_episodes, num_episodes_to_plot)

print("selected episodes:", target_episodes)


# =========================
# Evaluate selected episodes
# =========================

for ep in target_episodes:
    print("\n================================")
    print(f"Evaluating episode {ep}")
    print("================================")

    # Important: reset only once at the beginning of each episode
    if hasattr(policy, "reset"):
        policy.reset()
        print("Policy reset once at episode start.")

    episode_gt = []
    episode_pred = []

    indices = episode_to_indices[ep]

    for local_t, idx in enumerate(indices):
        sample = dataset[idx]

        gt, pred = predict_one_sample(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            sample=sample,
            device=device,
        )

        if local_t == 0:
            print("First GT:")
            print(gt)
            print("First Pred:")
            print(pred)
            print("GT shape:", gt.shape)
            print("Pred shape:", pred.shape)

        if pred.shape != gt.shape:
            print("WARNING: pred / gt shape mismatch")
            print("idx:", idx)
            print("gt shape:", gt.shape)
            print("pred shape:", pred.shape)
            continue

        episode_gt.append(gt)
        episode_pred.append(pred)

    episode_gt = torch.stack(episode_gt)
    episode_pred = torch.stack(episode_pred)

    print("episode_gt shape:", episode_gt.shape)
    print("episode_pred shape:", episode_pred.shape)

    print("GT mean:", episode_gt.mean(dim=0))
    print("Pred mean:", episode_pred.mean(dim=0))
    print("GT std:", episode_gt.std(dim=0))
    print("Pred std:", episode_pred.std(dim=0))

    action_dim = episode_gt.shape[1]

    fig, axes = plt.subplots(action_dim, 1, figsize=(16, 3 * action_dim), sharex=True)

    if action_dim == 1:
        axes = [axes]

    for d in range(action_dim):
        axes[d].plot(episode_gt[:, d].numpy(), label=f"GT dim {d}")
        axes[d].plot(episode_pred[:, d].numpy(), "--", label=f"Pred dim {d}")
        axes[d].set_ylabel(f"action {d}")
        axes[d].grid(True)
        axes[d].legend(loc="upper right")

    axes[-1].set_xlabel("Timestep within episode")
    fig.suptitle(f"Open-loop GT vs Pred - Episode {ep}", fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(
        plot_dir,
        f"episode_{ep}_gt_vs_pred_8dims_reset_once.png"
    )
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("Saved:", save_path)


print("\nDone.")
print("All plots saved to:", plot_dir)
