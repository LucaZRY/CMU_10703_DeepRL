 
#! python3

import argparse
import os
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np  # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PGConfig:
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    gamma: float = 0.99
    hidden: int = 256
    device: str = "cpu"


class PolicyGradient(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        lr_actor=1e-3,
        lr_critic=1e-3,
        mode="REINFORCE",
        n=0,
        gamma=0.99,
        device="cpu",
        hidden_layer_size=256,
    ):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode  # "REINFORCE", "REINFORCE_WITH_BASELINE", "A2C"
        self.n = n        # only used for A2C; ignored otherwise
        self.gamma = gamma

        self.device = torch.device(device)

        # -----------------
        # Networks
        # -----------------
        # Actor outputs logits; we will form a Categorical distribution from logits
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
        )

        # Critic outputs a scalar state-value (or baseline). Even if unused (REINFORCE),
        # we define it for a consistent interface.
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
        )

        # Init weights nicely (orthogonal helps for RL MLPs)
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

        self.actor.apply(init_layer)
        self.critic.apply(init_layer)

        # Separate optimizers per algorithm spec
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.to(self.device)

    # ------------- helpers -------------
    def dist(self, logits):
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, state, stochastic: bool):
        """
        state: np.ndarray (obs) or torch.Tensor [state_size]
        returns: action (int), logprob (torch.Tensor), value (torch.Tensor)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        logits = self.actor(state)
        value = self.critic(state).squeeze(-1)
        d = self.dist(logits)
        if stochastic:
            action = d.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        logp = d.log_prob(action)
        return int(action.item()), logp, value

    # ---------- returns / targets ----------
    def discounted_returns(self, rewards):
        """
        Compute episodic discounted returns G_t for REINFORCE variants.
        rewards: list[float] length T
        returns: torch.Tensor [T] on device
        """
        T = len(rewards)
        G = torch.zeros(T, dtype=torch.float32, device=self.device)
        running = torch.tensor(0.0, device=self.device)
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            G[t] = running
        return G

    def nstep_bootstrap_targets(self, rewards, values):
        """
        Compute N-step bootstrapped targets for A2C.
        rewards: list[float] length T
        values: torch.Tensor [T+1] where values[T] is 0 (terminal) or V(s_T) if bootstrapping allowed.
        returns: torch.Tensor [T] G_t^n
        """
        T = len(rewards)
        G = torch.zeros(T, dtype=torch.float32, device=self.device)
        n = self.n
        gamma = self.gamma
        # Precompute powers of gamma up to n
        gpow = torch.tensor([gamma ** i for i in range(n+1)], device=self.device, dtype=torch.float32)

        # Work backwards
        # Efficient rolling window sum: but clarity first
        for t in range(T):
            end = min(t + n, T)  # exclusive index for rewards sum
            steps = end - t
            # discounted sum of rewards from t to end-1
            ret = torch.tensor(0.0, device=self.device)
            for k in range(steps):
                ret += (gamma ** k) * rewards[t + k]
            # bootstrap if not beyond episode
            if t + n < T:
                ret += (gamma ** n) * values[t + n]
            G[t] = ret
        return G

    # ------------- train on a single episode -------------
    def train_on_episode(self, states, actions, rewards):
        """
        Perform a single update from one episode (entire episode as a minibatch).
        states: list[np.array] length T
        actions: list[int] length T
        rewards: list[float] length T
        """
        T = len(rewards)
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)

        # Forward pass for whole episode
        logits = self.actor(states_t)                      # [T, A]
        values = self.critic(states_t).squeeze(-1)         # [T]
        d = self.dist(logits)
        logprobs = d.log_prob(actions_t)                   # [T]

        if self.mode == "REINFORCE":
            targets = self.discounted_returns(rewards)     # G_t
            actor_loss = -(targets.detach() * logprobs).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            # No critic update
            critic_loss = torch.tensor(0.0, device=self.device)

        elif self.mode == "REINFORCE_WITH_BASELINE":
            targets = self.discounted_returns(rewards)     # G_t
            # Actor uses baseline but baseline should not get gradients through actor loss
            advantages = targets - values.detach()
            actor_loss = -(advantages * logprobs).mean()

            # Critic (baseline) regression to returns
            critic_loss = torch.mean((targets - values) ** 2)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        elif self.mode == "A2C":
            # Need V(s_{t+N}) for bootstrapping; compute values_plus with extra 0 at end
            values_plus = torch.cat([values, torch.tensor([0.0], device=self.device)])
            targets = self.nstep_bootstrap_targets(rewards, values_plus)  # G_t^(n)

            advantages = targets - values
            actor_loss = -(advantages.detach() * logprobs).mean()
            critic_loss = torch.mean(advantages ** 2)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "T": T,
            "G0": float(sum(rewards)),
        }

    # ------------- rollout -------------
    def rollout(self, env, max_steps=200, stochastic=True):
        total_reward = 0.0
        s, _ = env.reset()
        for _ in range(max_steps):
            a, _, _ = self.get_action(s, stochastic=stochastic)
            s, r, terminated, truncated, _ = env.step(a)
            total_reward += r
            if terminated or truncated:
                break
        return total_reward

    # ------------- train loop -------------
    def run(self, env, max_steps, num_episodes, evaluate_every=100, num_test_episodes=20, seed=None):
        """
        Train for num_episodes, evaluating every evaluate_every episodes with num_test_episodes
        returns: list[float] of evaluation means over time (length num_episodes // evaluate_every)
        """
        eval_means = []
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        for ep in range(1, num_episodes + 1):
            states, actions, rewards = [], [], []
            s, _ = env.reset()
            for _ in range(max_steps):
                a, logp, v = self.get_action(s, stochastic=True)
                s2, r, terminated, truncated, _ = env.step(a)

                states.append(s)
                actions.append(a)
                rewards.append(r)

                s = s2
                if terminated or truncated:
                    break

            # one update per episode
            self.train_on_episode(states, actions, rewards)

            # evaluation
            if ep % evaluate_every == 0:
                rets = [self.rollout(env, max_steps=max_steps, stochastic=False) for _ in range(num_test_episodes)]
                eval_means.append(float(np.mean(rets)))
        return eval_means


def graph_agents(
    graph_name,
    mode,
    env_name,
    max_steps,
    num_episodes,
    num_test_episodes,
    graph_every,
    num_runs,
    n_for_a2c=0,
    seed_base=0,
    device="cpu",
):
    print(f"Starting: {graph_name}")
    os.makedirs("./graphs", exist_ok=True)

    # For A2C annotate with N
    if mode == "A2C":
        graph_name = f"{graph_name}_{n_for_a2c}"

    # Collect D: shape [num_runs, num_snapshots]
    all_eval = []

    for run in range(num_runs):
        seed = seed_base + run * 100

        env = gym.make(env_name)
        try:
            env.reset(seed=seed)
        except TypeError:
            # Older gymnasium/gym versions
            env.reset()

        obs_space = env.observation_space
        act_space = env.action_space
        state_size = obs_space.shape[0]
        if hasattr(act_space, "n"):
            action_size = act_space.n
        else:
            raise ValueError("This implementation expects a discrete action space.")

        agent = PolicyGradient(
            state_size=state_size,
            action_size=action_size,
            lr_actor=1e-3,
            lr_critic=1e-3,
            mode=mode,
            n=n_for_a2c if mode == "A2C" else 0,
            gamma=0.99,
            device=device,
            hidden_layer_size=256,
        )

        # Train & evaluate
        eval_means = agent.run(
            env,
            max_steps=max_steps,
            num_episodes=num_episodes,
            evaluate_every=graph_every,
            num_test_episodes=num_test_episodes,
            seed=seed,
        )
        all_eval.append(eval_means)
        env.close()

    D = np.array(all_eval)  # shape [num_runs, num_snapshots]

    # Aggregate stats across runs per snapshot
    average_total_rewards = D.mean(axis=0)
    min_total_rewards = D.min(axis=0)
    max_total_rewards = D.max(axis=0)

    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    ax.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Total Reward")
    fig.savefig(f"./graphs/{graph_name}.png")
    plt.close(fig)
    print(f"Finished: {graph_name}")
    return D


def parse_args():
    mode_choices = ["REINFORCE", "REINFORCE_WITH_BASELINE", "A2C"]

    parser = argparse.ArgumentParser(description="Train an agent (10-703 HW1).")
    parser.add_argument(
        "--mode",
        type=str,
        default="REINFORCE",
        choices=mode_choices,
        help="Mode to run the agent in",
    )
    parser.add_argument("--n", type=int, default=0, help="The n to use for n-step A2C")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of IID runs (trials) to average over for graphs",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3500, help="Number of training episodes"
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=20,
        help="Number of test episodes per evaluation snapshot",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum steps per episode in the environment",
    )
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Gymnasium environment name"
    )
    parser.add_argument(
        "--graph_every", type=int, default=100, help="Evaluate every x episodes"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="torch device (cpu or cuda)"
    )
    parser.add_argument(
        "--seed_base", type=int, default=0, help="Base seed; each run adds an offset"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mode = args.mode
    graph_name = mode if mode != "A2C" else f"A2C"
    D = graph_agents(
        graph_name=graph_name,
        mode=mode,
        env_name=args.env_name,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes,
        num_test_episodes=args.num_test_episodes,
        graph_every=args.graph_every,
        num_runs=args.num_runs,
        n_for_a2c=args.n,
        seed_base=args.seed_base,
        device=args.device,
    )
    # Optionally save the raw matrix D for later analysis
    os.makedirs("./graphs", exist_ok=True)
    out_npy = f'./graphs/{graph_name if mode!="A2C" else f"A2C_{args.n}"}_D.npy'
    np.save(out_npy, D)
    print(f"Saved raw evaluation matrix to {out_npy}")


if __name__ == "__main__":
    main()
