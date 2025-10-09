# sac_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from buffer import Buffer
from policies import Actor, Critic


class SACAgent:
    """
    Soft Actor-Critic agent that matches PPO's interface pattern.
    Simplified to remove unnecessary abstraction layers.
    """
    
    def __init__(self, env_info, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, 
                 batch_size=128, update_every=1, buffer_size=100000, 
                 warmup_steps=5000, utd_ratio=1, device="cpu"):
        self.device = torch.device(device)
        
        # Environment info
        self.obs_dim = env_info["obs_dim"]
        self.act_dim = env_info["act_dim"]
        self.act_low = torch.as_tensor(env_info["act_low"], dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(env_info["act_high"], dtype=torch.float32, device=self.device)
        
        # SAC hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self.utd_ratio = utd_ratio
        
        # ================== Problem 3.1.1: SAC initialization ==================
        ### BEGIN STUDENT SOLUTION - 3.1.1 ###
        self.actor   = Actor(self.obs_dim, 
                             self.act_dim, 
                             self.act_low, 
                             self.act_high, 
                             hidden=(64, 64)).to(self.device)
        
        self.critic1 = Critic(self.obs_dim, 
                              self.act_dim, 
                              hidden=(64, 64)).to(self.device)
        
        self.critic2 = Critic(self.obs_dim, 
                              self.act_dim, 
                              hidden=(64, 64)).to(self.device)

        self.critic1_target = Critic(self.obs_dim, 
                                     self.act_dim, 
                                     hidden=(64, 64)).to(self.device)
        
        self.critic2_target = Critic(self.obs_dim, 
                                     self.act_dim, 
                                     hidden=(64, 64)).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        ### END STUDENT SOLUTION  -  3.1.1 ###
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        self._buffer = Buffer(
            size=buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=device
        )
        
        # Training state
        self.total_steps = 0
    
    def act(self, obs):
        """Return action info dict matching PPO's interface"""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist = self.actor(obs_t)
            action = dist.sample()
            
            # ---------------- Problem 3.5: Deterministic Action ----------------
            ### BEGIN STUDENT SOLUTION - 3.5 ###

            # action = self.actor(obs_t).mean_action
            
            ### END STUDENT SOLUTION  -  3.5 ###
            # Clamp to environment bounds
            action = torch.clamp(action, self.act_low, self.act_high)
            
            return {
                "action": action.squeeze(0).cpu().numpy()
            }
    
    def step(self, transition: Dict[str, Any]) -> Dict[str, float]:
        """
        Add transition to buffer and perform updates when ready.
        Matches PPO's step interface.
        """
        # Add to buffer using existing Buffer.add method
        obs_t = torch.as_tensor(transition["obs"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(transition["next_obs"], dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(transition["action"], dtype=torch.float32, device=self.device)
        
        self._buffer.add(
            obs=obs_t,
            next_obs=next_obs_t,
            action=action_t,
            log_probs=0.0,  # Not used in SAC
            reward=float(transition["reward"]),
            done=float(transition["done"]),
            value=0.0,  # Not used in SAC
            advantage=0.0,  # Not used in SAC
            curr_return=0.0,  # Not used in SAC
            iteration=0  # Not used in SAC
        )
        
        self.total_steps += 1
        
        # Check if we should update
        # ---------------- Problem 3.2: Environment Step ----------------
        ### BEGIN STUDENT SOLUTION - 3.2 ###
        # Only update if: finished warmup, buffer has enough samples, and it's an update step
        if (self.total_steps < self.warmup_steps):
            return {}
        if (self._buffer.size < self.batch_size):
            return {}
        if (self.total_steps % self.update_every != 0):
            return {}
        ### END STUDENT SOLUTION  -  3.2 ###
        
        # Perform SAC updates
        return self._perform_update()
    
    def _perform_update(self) -> Dict[str, float]:
        """Perform SAC updates and return stats"""
        all_stats = []
        
        # Perform multiple updates based on UTD ratio
        num_updates = max(1, self.utd_ratio)
        
        for _ in range(num_updates):
            # Sample batch from buffer
            batch = self._buffer.sample(self.batch_size)
            
            # Perform one SAC update step
            stats = self._sac_update_step(batch)
            all_stats.append(stats)
        
        # Average stats across updates
        if all_stats:
            return {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}
        else:
            return {}
    
    def _sac_update_step(self, batch) -> Dict[str, float]:
        """Single SAC update step"""
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        
        # entropy = 0.0 # placeholder
        # current_q1 = torch.zeros_like((actions.shape[0],)) # placeholder
        # current_q2 = torch.zeros_like((actions.shape[0],)) # placeholder
        # target_q = torch.zeros_like((actions.shape[0],)) # placeholder
        # actor_loss = torch.tensor(0.0) # placeholder
        
        # ---------------- Problem 3.1.2: Soft Bellman target ----------------
        ### BEGIN STUDENT SOLUTION - 3.1.2 ###
        with torch.no_grad():
            next_dist = self.actor(next_obs)
            next_actions = next_dist.rsample()
            next_logp = next_dist.log_prob(next_actions)
            if next_logp.dim() == 2:
                next_logp = next_logp.sum(-1, keepdim=True)
            else:
                next_logp = next_logp.unsqueeze(-1)
            next_logp = torch.clamp(next_logp, -20.0, 20.0)

            # Target critics
            q1_next = self.critic1_target(next_obs, next_actions)
            q2_next = self.critic2_target(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next)

            # y = r + γ(1-d) [ minQ' - α log π ]
            target_q = rewards.unsqueeze(-1) + self.gamma * (1.0 - dones.unsqueeze(-1)) * (min_q_next - self.alpha * next_logp)
            target_q = target_q.detach()

        ### END STUDENT SOLUTION  -  3.1.2 ###
        
        # ---------------- Problem 3.1.3: Critic update ----------------
        ### BEGIN STUDENT SOLUTION - 3.1.3 ###
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)

        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters()), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(list(self.critic2.parameters()), max_norm=1.0)

        self.critic_opt.step()
        ### END STUDENT SOLUTION  -  3.1.3 ###
        
        
        
        # ---------------- Problem 3.1.4: Actor update ----------------
        ### BEGIN STUDENT SOLUTION - 3.1.4 ###
        dist = self.actor(obs)
        new_actions = dist.rsample()
        logp = dist.log_prob(new_actions)
        if logp.dim() == 2:
            logp = logp.sum(-1, keepdim=True)
        else:
            logp = logp.unsqueeze(-1)
        logp = torch.clamp(logp, -20.0, 20.0)

        q1_pi = self.critic1(obs, new_actions)
        q2_pi = self.critic2(obs, new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # L_pi = E[ α log π(a|s) - min(Q1,Q2) ]
        actor_loss = (self.alpha * logp - min_q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        entropy = float((-logp).mean().item())

        ### END STUDENT SOLUTION  -  3.1.4 ###
        
        # ---------------- Problem 3.1.5: Target soft-updates ---------------
        ### BEGIN STUDENT SOLUTION - 3.1.5 ###
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        ### END STUDENT SOLUTION  -  3.1.5 ###
        
        # Return stats in format expected by runner
        return {
            "actor_loss": float(actor_loss.item()),
            "critic1_loss": float(nn.functional.mse_loss(current_q1, target_q).item()),
            "critic2_loss": float(nn.functional.mse_loss(current_q2, target_q).item()),
            "q1": float(current_q1.mean().item()),
            "q2": float(current_q2.mean().item()),
            "entropy": entropy
        }
    
    def _soft_update(self, local_model, target_model):
        """Soft update target network parameters"""
        # ---------------- Problem 3.1.5 Helper: Soft update implementation ----------------
        ### BEGIN STUDENT SOLUTION - 3.1.5 HELPER ###
        with torch.no_grad():
            for p, tp in zip(local_model.parameters(), target_model.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        ### END STUDENT SOLUTION  -  3.1.5 HELPER ###