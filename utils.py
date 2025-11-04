import os
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import gymnasium as gym
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import minari
from buffer import Buffer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_buffer_and_environments_for_task(
    minari_dataset_name: str = "mujoco/hopper/medium-v0",
    device="cpu",
):
    """
    Loads a Minari dataset, creates an evaluation environment, and populates a
    replay buffer with the dataset's transitions.

    Args:
        minari_dataset_name (str): The name of the Minari dataset to load.
        device (str): The device to store the buffer data on.

    Returns:
        tuple: A tuple containing:
            - Buffer: The replay buffer populated with offline data.
            - gym.Env: The evaluation environment.
    """
    dataset = None
    offline_buffer = None
    eval_env = None

    breakpoint()

    # ------ Problem 1.1: Loading the offline dataset into the replay buffer, creating the evaluation environment ------
    # You should reference the Minari documentation to load the offline dataset and create the evaluation environment.
    # https://minari.farama.org/content/basic_usage/
    #
    # Hint: when you load the dataset from Minari, it will give you episodes that you can iterate over.
    # However, the replay buffer stores transitions, so you need to get all the transitions for each episode and add them to the buffer.
    # If you are unsure how to get things from Minari objects (or any Python object really) a good first step is to enter a breakpoint
    # in pdb and inspect which fields are available with `p dir(object)`.

    ### BEGIN STUDENT SOLUTION - 1.1###
    raise NotImplementedError("Student exercise: complete the code for Problem 1.1")
    ### END STUDENT SOLUTION - 1.1###

    assert (
        offline_buffer.size == dataset.total_steps
    ), "The buffer size should be equal to the total number of steps in the dataset"
    assert eval_env is not None, "The evaluation environment should not be None"

    return offline_buffer, eval_env


@torch.no_grad()
def greedy_action(policy, obs_t):
    """Get deterministic action from policy/actor"""
    if hasattr(policy, "forward"):
        # This is an ActorCritic (PPO) - returns (dist, value)
        result = policy(obs_t)
        if isinstance(result, tuple):
            dist, _ = result
        else:
            dist = result
    else:
        # This is just an Actor (SAC) - returns dist
        dist = policy(obs_t)

    if hasattr(dist, "mean_action"):  # continuous
        a = dist.mean_action
    else:  # discrete
        a = torch.argmax(dist.logits, dim=-1)
    return a


def _to_env_action(env, action_tensor):
    if isinstance(env.action_space, gym.spaces.Box):
        a = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        return np.clip(a, env.action_space.low, env.action_space.high)
    else:
        return int(action_tensor.item())


def evaluate_policy(agent, env, episodes=10, seed=42, device="cpu"):
    """Evaluate agent performance - works for both PPO and SAC"""
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = truncated = False
        ep_r = 0.0
        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            # Get the policy/actor from agent
            if hasattr(agent, "actor"):
                policy = agent.actor  # SAC
            elif hasattr(agent, "policy"):
                policy = agent.policy  # PPO
            else:
                raise ValueError(
                    f"Agent {type(agent)} has no 'actor' or 'policy' attribute"
                )

            a = greedy_action(policy, obs_t)
            a_env = _to_env_action(env, a)
            obs, r, done, truncated, _ = env.step(a_env)
            ep_r += r
        scores.append(ep_r)
    env.close()
    return float(np.mean(scores)), float(np.std(scores))


def record_eval_video(
    agent,
    env,
    video_dir="videos",
    video_name="eval",
    seed=None,
    episodes=1,
    device="cpu",
):
    """Record evaluation video - works for both PPO and SAC"""
    os.makedirs(video_dir, exist_ok=True)

    frames = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep if seed is not None else None)
        done = truncated = False
        frames.append(env.render())
        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            # Get the policy/actor from agent
            if hasattr(agent, "actor"):
                policy = agent.actor  # SAC
            elif hasattr(agent, "policy"):
                policy = agent.policy  # PPO
            else:
                raise ValueError(
                    f"Agent {type(agent)} has no 'actor' or 'policy' attribute"
                )

            a = greedy_action(policy, obs_t)
            a_env = _to_env_action(env, a)
            obs, _, done, truncated, _ = env.step(a_env)
            frames.append(env.render())

    env.close()

    video_path = os.path.join(video_dir, f"{video_name}.mp4")
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(video_path, codec="libx264", logger=None)
    return video_path


def detect_agent_type(log):
    """Detect whether this is PPO or SAC based on log keys"""
    ppo_keys = {"policy_loss", "value_loss", "entropy", "kl", "clipfrac"}
    sac_keys = {"actor_loss", "critic1_loss", "critic2_loss", "alpha", "q1", "q2"}
    td3_keys = {
        "ddpg_policy_loss",
        "bc_regularization_loss",
        "cql_loss",
        "in_distribution_q_pred",
    }

    log_keys = set(log.keys())

    if ppo_keys.intersection(log_keys):
        return "ppo"
    if td3_keys.intersection(log_keys):
        return "td3"
    elif sac_keys.intersection(log_keys):
        return "sac"
    else:
        # Default fallback
        return "unknown"


def plot_curves(log, out_path="training_curves.png"):
    """
    Universal plotting function that adapts to PPO or SAC based on log contents
    """
    agent_type = detect_agent_type(log)

    if agent_type == "ppo":
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("PPO Training Progress", fontsize=16)
        plot_ppo_metrics(log, axes)
    elif agent_type == "sac":
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("SAC Training Progress", fontsize=16)
        plot_sac_metrics(log, axes)
    elif agent_type == "td3":
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle("TD3 Training Progress", fontsize=16)
        plot_td3_metrics(log, axes)
    else:
        # Fallback: just plot what we can
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("Training Progress", fontsize=16)
        plot_basic_metrics(log, axes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {out_path}")


def plot_ppo_metrics(log, axes):
    """Plot PPO-specific metrics"""

    # Helper function to get x-axis values
    def get_x_values(data_key):
        if data_key == "episodic_return":
            return log.get("steps", list(range(len(log.get(data_key, [])))))[
                : len(log.get(data_key, []))
            ]
        else:
            # For loss metrics, estimate step values
            steps = log.get("steps", [])
            loss_data = log.get(data_key, [])
            if len(loss_data) == 0:
                return []
            total_steps = steps[-1] if steps else len(loss_data)
            return np.linspace(0, total_steps, len(loss_data))

    # Plot episodic returns
    if "episodic_return" in log and len(log["episodic_return"]) > 0:
        x_vals = get_x_values("episodic_return")
        axes[0, 0].plot(x_vals, log["episodic_return"], "b-", alpha=0.7, linewidth=1.5)
        axes[0, 0].set_title("Episode Returns")
        axes[0, 0].set_xlabel("Environment Steps")
        axes[0, 0].set_ylabel("Return")
        axes[0, 0].grid(True, alpha=0.3)

        # Add moving average
        if len(log["episodic_return"]) > 10:
            window = min(50, len(log["episodic_return"]) // 10)
            moving_avg = np.convolve(
                log["episodic_return"], np.ones(window) / window, mode="valid"
            )
            ma_x = x_vals[window - 1 : len(moving_avg) + window - 1]
            axes[0, 0].plot(
                ma_x, moving_avg, "r-", linewidth=2, alpha=0.8, label=f"MA({window})"
            )
            axes[0, 0].legend()

    # Plot PPO-specific losses
    ppo_metrics = [
        ("loss", "Total Loss", (0, 1)),
        ("policy_loss", "Policy Loss", (0, 2)),
        ("value_loss", "Value Loss", (1, 0)),
        ("entropy", "Entropy", (1, 1)),
        ("kl", "KL Divergence", (1, 2)),
    ]

    for key, title, (i, j) in ppo_metrics:
        if key in log and len(log[key]) > 0:
            x_vals = get_x_values(key)
            if len(x_vals) > 0:
                axes[i, j].plot(x_vals, log[key], linewidth=1.5, alpha=0.8)
                axes[i, j].set_title(title)
                axes[i, j].set_xlabel("Environment Steps")
                axes[i, j].set_ylabel(title)
                axes[i, j].grid(True, alpha=0.3)

                # Add KL divergence reference line
                if key == "kl":
                    axes[i, j].axhline(
                        y=0.01, color="r", linestyle="--", alpha=0.5, label="Target KL"
                    )
                    axes[i, j].legend()


def plot_sac_metrics(log, axes):
    """Plot SAC-specific metrics"""

    def get_update_steps(data_key):
        """Get x-axis values for update-based metrics"""
        data = log.get(data_key, [])
        if len(data) == 0:
            return []
        # For SAC, updates happen frequently, so just use update index
        return list(range(len(data)))

    # Plot episodic returns (same as PPO)
    if "episodic_return" in log and len(log["episodic_return"]) > 0:
        steps = log.get("steps", list(range(len(log["episodic_return"]))))[
            : len(log["episodic_return"])
        ]
        axes[0, 0].plot(steps, log["episodic_return"], "b-", alpha=0.7, linewidth=1.5)
        axes[0, 0].set_title("Episode Returns")
        axes[0, 0].set_xlabel("Environment Steps")
        axes[0, 0].set_ylabel("Return")
        axes[0, 0].grid(True, alpha=0.3)

        # Add moving average
        if len(log["episodic_return"]) > 10:
            window = min(50, len(log["episodic_return"]) // 10)
            moving_avg = np.convolve(
                log["episodic_return"], np.ones(window) / window, mode="valid"
            )
            ma_x = steps[window - 1 : len(moving_avg) + window - 1]
            axes[0, 0].plot(
                ma_x, moving_avg, "r-", linewidth=2, alpha=0.8, label=f"MA({window})"
            )
            axes[0, 0].legend()

    # Actor loss
    if "actor_loss" in log and len(log["actor_loss"]) > 0:
        x_vals = get_update_steps("actor_loss")
        axes[0, 1].plot(x_vals, log["actor_loss"], "g-", linewidth=1.5, alpha=0.8)
        axes[0, 1].set_title("Actor Loss")
        axes[0, 1].set_xlabel("Updates")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].grid(True, alpha=0.3)

    # Critic losses
    if "critic1_loss" in log and "critic2_loss" in log:
        if len(log["critic1_loss"]) > 0 and len(log["critic2_loss"]) > 0:
            x_vals = get_update_steps("critic1_loss")
            axes[0, 2].plot(
                x_vals,
                log["critic1_loss"],
                "r-",
                linewidth=1.5,
                alpha=0.8,
                label="Critic 1",
            )
            axes[0, 2].plot(
                x_vals,
                log["critic2_loss"],
                "orange",
                linewidth=1.5,
                alpha=0.8,
                label="Critic 2",
            )
            axes[0, 2].set_title("Critic Losses")
            axes[0, 2].set_xlabel("Updates")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

    # Q-values
    if "q1" in log and "q2" in log:
        if len(log["q1"]) > 0 and len(log["q2"]) > 0:
            x_vals = get_update_steps("q1")
            axes[1, 0].plot(
                x_vals, log["q1"], "b-", linewidth=1.5, alpha=0.8, label="Q1"
            )
            axes[1, 0].plot(
                x_vals, log["q2"], "purple", linewidth=1.5, alpha=0.8, label="Q2"
            )
            axes[1, 0].set_title("Q-Values")
            axes[1, 0].set_xlabel("Updates")
            axes[1, 0].set_ylabel("Q-Value")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

    # Alpha (entropy coefficient) - now includes interpretation
    if "alpha" in log and len(log["alpha"]) > 0:
        x_vals = get_update_steps("alpha")
        axes[1, 1].plot(x_vals, log["alpha"], "m-", linewidth=1.5, alpha=0.8)
        axes[1, 1].set_title("Alpha (Entropy Regularization)")
        axes[1, 1].set_xlabel("Updates")
        axes[1, 1].set_ylabel("Alpha")
        axes[1, 1].grid(True, alpha=0.3)

        # Add explanation
        current_alpha = log["alpha"][-1] if log["alpha"] else 0.2
        axes[1, 1].text(
            0.02,
            0.98,
            f"Current α={current_alpha:.3f}\nHigher α = More Exploration",
            transform=axes[1, 1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
        )

    # Evaluation results (if available)
    if "eval_mean" in log and "eval_steps" in log and len(log["eval_mean"]) > 0:
        eval_std = log.get("eval_std", [0] * len(log["eval_mean"]))
        axes[1, 2].errorbar(
            log["eval_steps"],
            log["eval_mean"],
            yerr=eval_std,
            marker="o",
            linewidth=2,
            alpha=0.8,
        )
        axes[1, 2].set_title("Evaluation Returns")
        axes[1, 2].set_xlabel("Environment Steps")
        axes[1, 2].set_ylabel("Mean Return")
        axes[1, 2].grid(True, alpha=0.3)


def plot_td3_metrics(log, axes):
    """Plot TD3-specific metrics"""

    def get_update_steps(data_key):
        """Get x-axis values for update-based metrics"""
        data = log.get(data_key, [])
        if len(data) == 0:
            return []
        return list(range(len(data)))

    # Plot evaluation returns
    if "eval_mean" in log and "eval_steps" in log and len(log["eval_mean"]) > 0:
        ax = axes[0, 0]
        eval_std = log.get("eval_std", [0] * len(log["eval_mean"]))
        ax.errorbar(
            log["eval_steps"],
            log["eval_mean"],
            yerr=eval_std,
            marker="o",
            linewidth=2,
            alpha=0.8,
        )
        ax.set_title("Evaluation Returns")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Return")
        ax.grid(True, alpha=0.3)

    # Actor loss (can be 'actor_loss' or 'ddpg_policy_loss')
    actor_loss_key = None
    if "ddpg_policy_loss" in log and len(log["ddpg_policy_loss"]) > 0:
        actor_loss_key = "ddpg_policy_loss"
    elif "actor_loss" in log and len(log["actor_loss"]) > 0:
        actor_loss_key = "actor_loss"

    if actor_loss_key:
        ax = axes[0, 1]
        x_vals = get_update_steps(actor_loss_key)
        ax.plot(x_vals, log[actor_loss_key], "g-", linewidth=1.5, alpha=0.8)
        ax.set_title("Actor Loss")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    # Critic losses
    if "critic1_loss" in log and "critic2_loss" in log:
        if len(log["critic1_loss"]) > 0 and len(log["critic2_loss"]) > 0:
            ax = axes[0, 2]
            x_vals = get_update_steps("critic1_loss")
            ax.plot(
                x_vals,
                log["critic1_loss"],
                "r-",
                linewidth=1.5,
                alpha=0.8,
                label="Critic 1",
            )
            ax.plot(
                x_vals,
                log["critic2_loss"],
                "orange",
                linewidth=1.5,
                alpha=0.8,
                label="Critic 2",
            )
            ax.set_title("Critic Losses")
            ax.set_xlabel("Updates")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Q-values
    if "q1" in log and "q2" in log:
        if len(log["q1"]) > 0 and len(log["q2"]) > 0:
            ax = axes[1, 0]
            x_vals = get_update_steps("q1")
            ax.plot(x_vals, log["q1"], "b-", linewidth=1.5, alpha=0.8, label="Q1")
            ax.plot(x_vals, log["q2"], "purple", linewidth=1.5, alpha=0.8, label="Q2")
            ax.set_title("Q-Values")
            ax.set_xlabel("Updates")
            ax.set_ylabel("Q-Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # BC regularization loss
    if "bc_regularization_loss" in log and len(log["bc_regularization_loss"]) > 0:
        ax = axes[1, 1]
        x_vals = get_update_steps("bc_regularization_loss")
        ax.plot(x_vals, log["bc_regularization_loss"], linewidth=1.5, alpha=0.8)
        ax.set_title("BC Regularization Loss")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    # BC regularization weight
    if "bc_regularization_weight" in log and len(log["bc_regularization_weight"]) > 0:
        ax = axes[1, 2]
        x_vals = get_update_steps("bc_regularization_weight")
        ax.plot(x_vals, log["bc_regularization_weight"], linewidth=1.5, alpha=0.8)
        ax.set_title("BC Regularization Weight")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3)

    # CQL Loss
    if "cql_loss" in log and len(log["cql_loss"]) > 0:
        ax = axes[2, 0]
        x_vals = get_update_steps("cql_loss")
        ax.plot(x_vals, log["cql_loss"], linewidth=1.5, alpha=0.8)
        ax.set_title("CQL Loss")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    # CQL Q-values for current and random actions
    if "cql_current_actions" in log and "cql_random_actions" in log:
        if len(log["cql_current_actions"]) > 0 and len(log["cql_random_actions"]) > 0:
            ax = axes[2, 1]
            x_vals = get_update_steps("cql_current_actions")
            ax.plot(
                x_vals,
                log["cql_current_actions"],
                linewidth=1.5,
                alpha=0.8,
                label="Current Actions",
            )
            ax.plot(
                x_vals,
                log["cql_random_actions"],
                linewidth=1.5,
                alpha=0.8,
                label="Random Actions",
            )
            ax.set_title("CQL Action Q-Values")
            ax.set_xlabel("Updates")
            ax.set_ylabel("Q-Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # In-distribution Q prediction
    if "in_distribution_q_pred" in log and len(log["in_distribution_q_pred"]) > 0:
        ax = axes[2, 2]
        x_vals = get_update_steps("in_distribution_q_pred")
        ax.plot(x_vals, log["in_distribution_q_pred"], linewidth=1.5, alpha=0.8)
        ax.set_title("In-Distribution Q-Value Pred")
        ax.set_xlabel("Updates")
        ax.set_ylabel("Q-Value")
        ax.grid(True, alpha=0.3)


def plot_basic_metrics(log, axes):
    """Fallback plotting for unknown agent types"""

    # Plot episodic returns if available
    if "episodic_return" in log and len(log["episodic_return"]) > 0:
        steps = log.get("steps", list(range(len(log["episodic_return"]))))[
            : len(log["episodic_return"])
        ]
        axes[0].plot(steps, log["episodic_return"], "b-", alpha=0.7, linewidth=1.5)
        axes[0].set_title("Episode Returns")
        axes[0].set_xlabel("Environment Steps")
        axes[0].set_ylabel("Return")
        axes[0].grid(True, alpha=0.3)

    # Plot any loss-like metrics
    axes[1].set_title("Loss Metrics")
    axes[1].set_xlabel("Updates")
    axes[1].set_ylabel("Loss Value")
    axes[1].grid(True, alpha=0.3)

    loss_keys = [k for k in log.keys() if "loss" in k.lower() and len(log[k]) > 0]
    for key in loss_keys[:5]:  # Limit to 5 metrics to avoid clutter
        if len(log[key]) > 0:
            axes[1].plot(
                range(len(log[key])), log[key], label=key, alpha=0.8, linewidth=1.5
            )

    if loss_keys:
        axes[1].legend()
        axes[1].set_yscale("symlog", linthresh=1e-3)
