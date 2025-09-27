# ppo_agent.py
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from buffer import Buffer
from policies import ActorCritic

class PPOAgent:
    def __init__(self, env_info, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_coef=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                 update_epochs=10, minibatch_size=64, rollout_steps=4096, device="cpu"):
        self.device = torch.device(device)
        policy = ActorCritic(
            env_info["obs_dim"],
            env_info["act_dim"],
            env_info["act_low"],
            env_info["act_high"],
            hidden=(64, 64),
        )
        self.actor = policy.to(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        
        # PPO with KL penalty parameters
        self.beta = .5  # Initial KL penalty coefficient
        self.target_kl = 0.01  # Target KL divergence
        
        # Internal state for rollout collection
        self._curr_policy_rollout = []
        self._rollout_buffer = Buffer(
            size=rollout_steps*50,
            obs_dim=policy.obs_dim,
            act_dim=policy.act_dim,
            device=device
        )
        self._steps_collected_with_curr_policy = 0
        self._policy_iteration = 1
    
    def act(self, obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist, value = self.actor(obs_t)
            action = dist.sample()      
            log_prob = dist.log_prob(action)
            
            return {
                "action": action.squeeze(0).cpu().numpy(),
                "log_prob": float(log_prob.squeeze(0).item()),
                "value": float(value.squeeze(0).item())
            }

    def step(self, transition: Dict[str, Any]) -> Dict[str, float]:
        """
        PPO-specific step: collect transitions until rollout is full, then update.
        
        transition should contain:
        - obs, action, reward, next_obs, done, truncated
        - log_prob, value (from act() call)
        """
        # Add to current rollout
        self._curr_policy_rollout.append(transition.copy())
        self._steps_collected_with_curr_policy += 1
        stop = transition['done'] or transition['truncated']
        ret = {}
        # ---------------- Problem 1.3.1: PPO Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.3.1 ###

        ### END STUDENT SOLUTION - 1.3.1 ###

        return ret  # Leave this as an empty dictionary if no update is performed

    def _perform_update(self) -> Dict[str, float]:
        """Perform PPO update using collected rollout"""
        all_stats = []

        # To log metrics correctly, make sure you have the following lines in this function
        # loss, stats = self._ppo_loss(minibatch)
        # all_stats.append(stats)
        
        # ---------------- Problem 1.3.2: PPO Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.3.2 ###

        ### EXPERIMENT 1.6 CODE ###

        ### EXPERIMENT 1.6 CODE END ###
    
        ### END STUDENT SOLUTION - 1.3.2 ###
        
        # ---------------- Problem 1.4.2: KL Divergence Beta Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.4.2 ###

        ### END STUDENT SOLUTION - 1.4.2 ###
        
        if all_stats:
            return {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}
        else:
            return {}
        
    def _compute_gae(self, rollout) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rollout)
        rewards = np.array([t["reward"] for t in rollout])
        values = np.array([t["value"] for t in rollout])
        dones = np.array([t["done"] for t in rollout])  # Get done flag for each timestep
        
        # Get the final value for bootstrap
        next_obs = rollout[-1]["next_obs"]
        with torch.no_grad():
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, final_v = self.actor(obs_t)
            final_v = float(final_v.squeeze(0).item())
        
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        # ---------------- Problem 1.2: Compute GAE ----------------
        ### BEGIN STUDENT SOLUTION - 1.2 ###
        
        ### END STUDENT SOLUTION - 1.2 ###
        
        return advantages, returns
    
    def _ppo_loss(self, batch):
        """Standard PPO loss computation"""
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Forward pass
        dist, values = self.actor(obs)
        log_probs = dist.log_prob(actions)
        if log_probs.ndim > 1:
            log_probs = log_probs.sum(dim=-1)
            
        policy_loss = 0 # Placeholder
        total_loss = 0 # Placeholder
        ratio = 0 # Placeholder

        # ---------------- Problem 1.4.2: KL Divergence Policy Loss ----------------
        ### BEGIN STUDENT SOLUTION - 1.4.2 ###

        ### END STUDENT SOLUTION - 1.4.2 ###
        
        # ---------------- Problem 1.1.1: PPO Clipped Surrogate Objective Loss ----------------
        ### BEGIN STUDENT SOLUTION - 1.1.1 ###

        ### END STUDENT SOLUTION - 1.1.1 ###
        
        
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)

        entropy_loss = 0 # Placeholder
        value_loss = 0 # Placeholder

        # ---------------- Problem 1.1.2: PPO Total Loss (Include Entropy Bonus and Value Loss) ----------------
        ### BEGIN STUDENT SOLUTION - 1.1.2 ###

        ### END STUDENT SOLUTION - 1.1.2 ###

        # Stats
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
        
        return total_loss, {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(-entropy_loss.item()),
            "kl": float(approx_kl.item()),
            "clipfrac": float(clipfrac.item()),
        }
        
    def _prepare_batch(self, advantages, returns):
        """Collate the current rollout into a batch for the buffer"""
        obs = torch.stack([torch.as_tensor(t["obs"], dtype=torch.float32) for t in self._curr_policy_rollout])
        next_obs = torch.stack([torch.as_tensor(t["next_obs"], dtype=torch.float32) for t in self._curr_policy_rollout])
        actions = torch.stack([torch.as_tensor(t["action"], dtype=torch.float32) for t in self._curr_policy_rollout])
        log_probs = torch.tensor([t["log_prob"] for t in self._curr_policy_rollout], dtype=torch.float32)
        values = torch.tensor([t["value"] for t in self._curr_policy_rollout], dtype=torch.float32)
        rewards = torch.tensor([t["reward"] for t in self._curr_policy_rollout], dtype=torch.float32)
        
        return {
            "obs": obs.to(self.device),
            "next_obs": next_obs.to(self.device),
            "actions": actions.to(self.device),
            "log_probs": log_probs.to(self.device),
            "rewards": rewards.to(self.device),
            "values": values.to(self.device),
            "dones": torch.tensor([t["done"] for t in self._curr_policy_rollout], dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            "iteration": torch.full((len(self._curr_policy_rollout),), self._policy_iteration, dtype=torch.int32, device=self.device)
        }