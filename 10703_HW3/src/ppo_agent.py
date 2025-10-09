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
        if stop:
            advantages, returns = self._compute_gae(self._curr_policy_rollout)

            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            batch = self._prepare_batch(advantages, returns)
            self._last_batch = batch 

        if self._steps_collected_with_curr_policy >= self.rollout_steps:
            ret = self._perform_update()
            self._curr_policy_rollout = []
            self._steps_collected_with_curr_policy = 0
            self._policy_iteration += 1
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
        assert hasattr(self, "_last_batch"), "No prepared batch. Collect a rollout first."
        batch = self._last_batch
        N = batch["obs"].shape[0]

        for _ in range(self.update_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]
                minibatch = {
                    "obs": batch["obs"][idx],
                    "actions": batch["actions"][idx],
                    "log_probs": batch["log_probs"][idx],
                    "advantages": batch["advantages"][idx],
                    "returns": batch["returns"][idx],
                }

                self.optimizer.zero_grad(set_to_none=True)
                loss, stats = self._ppo_loss(minibatch)
                all_stats.append(stats)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()
        ### EXPERIMENT 1.6 CODE ###

        full_offpolicy = False
        half_offpolicy = True

        batch = self._last_batch
        curr = {k: batch[k] for k in ("obs","actions","log_probs","advantages","returns")}
        N_curr = curr["obs"].shape[0]

        if not hasattr(self, "_ppo_replay"):
            from collections import deque
            self._ppo_replay = deque(maxlen=64)

        self._ppo_replay.append({k: v.detach().clone() for k, v in curr.items()})

        def _concat_batches(batches):
            out = {}
            for k in ("obs","actions","log_probs","advantages","returns"):
                out[k] = torch.cat([b[k] for b in batches], dim=0)
            return out

        big = _concat_batches(list(self._ppo_replay))  # ALL data so far (including current)
        N_big = big["obs"].shape[0]

        # ---- Off-policy sampling variants ----
        for _ in range(self.update_epochs):
            if full_offpolicy:
                # ===== 1.6.1 FULL OFF-POLICY: sample only from BIG replay =====
                perm = torch.randperm(N_big, device=self.device)
                for start in range(0, N_big, self.minibatch_size):
                    idx = perm[start:start + self.minibatch_size]
                    mb = {k: big[k][idx] for k in curr.keys()}
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, stats = self._ppo_loss(mb)
                    all_stats.append(stats)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            elif half_offpolicy:
                # ===== 1.6.2 HALF OFF-POLICY: half from CURRENT, half from BIG =====
                half = self.minibatch_size // 2
                perm_curr = torch.randperm(N_curr, device=self.device)
                perm_big  = torch.randperm(N_big,  device=self.device)

                max_iter = max(
                    (N_curr + half - 1) // half,
                    (N_big  + half - 1) // half
                )
                for it in range(max_iter):
                    i1 = perm_curr[it*half:(it+1)*half]
                    i2 = perm_big[it*half:(it+1)*half]

                    # top-up if one pool runs short so batch size stays constant
                    if i1.numel() < half:
                        i1 = torch.cat([i1, torch.randint(0, N_curr, (half - i1.numel(),), device=self.device)], dim=0)
                    if i2.numel() < half:
                        i2 = torch.cat([i2, torch.randint(0, N_big,  (half - i2.numel(),), device=self.device)], dim=0)

                    mb = {}
                    for k in curr.keys():
                        mb[k] = torch.cat([curr[k][i1], big[k][i2]], dim=0)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss, stats = self._ppo_loss(mb)
                    all_stats.append(stats)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            else:
                # ===== On-policy fallback (current rollout only) =====
                perm = torch.randperm(N_curr, device=self.device)
                for start in range(0, N_curr, self.minibatch_size):
                    idx = perm[start:start + self.minibatch_size]
                    mb = {k: curr[k][idx] for k in curr.keys()}
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, stats = self._ppo_loss(mb)
                    all_stats.append(stats)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer.step()

        ### EXPERIMENT 1.6 CODE END ###
    
        ### END STUDENT SOLUTION - 1.3.2 ###
        
        # ---------------- Problem 1.4.2: KL Divergence Beta Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.4.2 ###

        # if all_stats:
        #     avg_kl = float(np.mean([s["kl"] for s in all_stats]))
        #     if avg_kl > 2.0 * self.target_kl:
        #         self.beta *= 1.5
        #     elif avg_kl < 0.5 * self.target_kl:
        #         self.beta /= 1.5
        #     self.beta = float(np.clip(self.beta, 1e-8, 10.0))
 
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
        values_ext = np.append(values.astype(np.float32), np.float32(final_v))
        advantages = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])  
            delta = rewards[t] + self.gamma * values_ext[t + 1] * nonterminal - values_ext[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values_ext[:-1]
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
        # ratio = torch.exp(log_probs - old_log_probs)              
        # approx_kl = (old_log_probs - log_probs).mean()         
        # policy_loss = -(ratio * advantages).mean() + self.beta * approx_kl
        ### END STUDENT SOLUTION - 1.4.2 ###
        
        # ---------------- Problem 1.1.1: PPO Clipped Surrogate Objective Loss ----------------
        ### BEGIN STUDENT SOLUTION - 1.1.1 ###
        ratio       = torch.exp(log_probs - old_log_probs)
        unclipped   = ratio * advantages
        clipped     = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        total_loss  = policy_loss
        ### END STUDENT SOLUTION - 1.1.1 ###
        
        
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)

        entropy_loss = 0 # Placeholder
        value_loss = 0 # Placeholder

        # ---------------- Problem 1.1.2: PPO Total Loss (Include Entropy Bonus and Value Loss) ----------------
        ### BEGIN STUDENT SOLUTION - 1.1.2 ###
        entropy_loss = - self.ent_coef * entropy.mean()
        value_loss   = (returns - values.squeeze(-1))**2
        value_loss   = value_loss.mean()
        total_loss   = policy_loss + self.vf_coef * value_loss + entropy_loss
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