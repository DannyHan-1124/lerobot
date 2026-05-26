import torch
import numpy as np

from lerobot.policies import Policy
from lerobot.envs import make_env

# -----------------------------
# CONFIG (matches your CLI)
# -----------------------------
REPO_ID = "Danny1124/pi05_run1"
TASK = "libero_object"
NUM_EPISODES = 20
MAX_STEPS = 280

# -----------------------------
# Load policy (IMPORTANT: includes preprocessors)
# -----------------------------
policy = Policy.from_pretrained(REPO_ID)
policy.eval()

# -----------------------------
# Create environment
# -----------------------------
env = make_env(
    "libero",
    task=TASK
)

successes = []
episode_rewards = []

# -----------------------------
# Evaluation loop
# -----------------------------
for ep in range(NUM_EPISODES):
    obs = env.reset()

    done = False
    total_reward = 0
    step = 0

    while not done and step < MAX_STEPS:

        # IMPORTANT: this applies preprocessor internally
        action = policy.select_action(obs)

        # convert if needed
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()

        obs, reward, done, info = env.step(action)

        total_reward += reward
        step += 1

    # LIBERO-style success flag
    success = info.get("success", False) if isinstance(info, dict) else False

    successes.append(success)
    episode_rewards.append(total_reward)

    print(f"Episode {ep}: reward={total_reward:.3f}, success={success}")

# -----------------------------
# Summary
# -----------------------------
success_rate = np.mean(successes)

print("\n===== EVAL SUMMARY =====")
print(f"Success rate: {success_rate:.3f}")
print(f"Mean reward: {np.mean(episode_rewards):.3f}")
print(f"Std reward: {np.std(episode_rewards):.3f}")
