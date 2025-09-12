#! python3

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np  # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


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
    ):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            # Output action probabilities for a categorical policy
            nn.Softmax(dim=-1),
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            # Scalar state-value baseline / critic
            nn.Linear(hidden_layer_size, 1),
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        self.to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        # END STUDENT SOLUTION

    def forward(self, state):
        return (self.actor(state), self.critic(state))

    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.actor(state_t).squeeze(0)
            if stochastic:
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample().item()
            else:
                action = torch.argmax(probs).item()
        return action
        # END STUDENT SOLUTION

    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        # rewards_tensor: (T,)
        # values: (T,) predicted V(s_t)
        T = rewards_tensor.shape[0]
        n = self.n
        gamma = self.gamma

        targets = torch.zeros(T, dtype=torch.float32, device=rewards_tensor.device)
        for t in range(T):
            end = min(t + n, T)
            # discounted rewards from t to end-1
            G = torch.zeros(1, device=rewards_tensor.device).squeeze()
            discount = 1.0
            for k in range(t, end):
                G = G + discount * rewards_tensor[k]
                discount *= gamma
            # bootstrap if within episode horizon
            if end < T:
                G = G + (discount) * values[end]
            # else Vend = 0 as per Algorithm 3
            targets[t] = G
        return targets
        # END STUDENT SOLUTION

    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION
        # Convert to tensors
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(np.array(rewards, dtype=np.float32), dtype=torch.float32, device=self.device)

        T = states_t.shape[0]

        # Policy (probs) and value predictions
        probs = self.actor(states_t)                            # (T, A), sums to 1
        log_probs = torch.log(probs.clamp_min(1e-8))            # stable log
        selected_log_probs = log_probs.gather(1, actions_t.view(-1, 1)).squeeze(1)  # (T,)

        values = self.critic(states_t).squeeze(-1)              # (T,)

        # Compute returns / targets and advantages depending on mode
        if self.mode == "REINFORCE":
            # Monte-Carlo returns
            returns = torch.zeros(T, dtype=torch.float32, device=self.device)
            G = 0.0
            for t in reversed(range(T)):
                G = rewards_t[t] + self.gamma * G
                returns[t] = G
            advantages = returns.detach()
            actor_loss = -(advantages * selected_log_probs).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # no critic update in vanilla REINFORCE
            return actor_loss.item()

        elif self.mode == "REINFORCE_WITH_BASELINE":
            # Monte-Carlo returns and learned baseline b_ω(s) ~ V(s)
            returns = torch.zeros(T, dtype=torch.float32, device=self.device)
            G = 0.0
            for t in reversed(range(T)):
                G = rewards_t[t] + self.gamma * G
                returns[t] = G

            advantages = (returns - values.detach())
            actor_loss = -(advantages * selected_log_probs).mean()

            critic_loss = F.mse_loss(values, returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            return (actor_loss.item(), critic_loss.item())

        elif self.mode == "A2C":
            # N-step bootstrapped targets
            with torch.no_grad():
                targets = self.calculate_n_step_bootstrap(rewards_t, values)

            advantages = (targets - values.detach())
            actor_loss = -(advantages * selected_log_probs).mean()

            critic_loss = F.mse_loss(values, targets)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            return (actor_loss.item(), critic_loss.item())

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        # END STUDENT SOLUTION

    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        stochastic = True if train else False

        for _ in range(num_episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            states, actions, rewards = [], [], []

            for _step in range(max_steps):
                action = self.get_action(obs, stochastic=stochastic)
                next_obs, reward, terminated, truncated, info = env.step(action)

                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                ep_reward += reward

                obs = next_obs
                if terminated or truncated:
                    break

            if train:
                self.train(states, actions, rewards)

            total_rewards.append(ep_reward)
        # END STUDENT SOLUTION
        return total_rewards


def graph_agents(
    graph_name,
    agents,
    env,
    max_steps,
    num_episodes,
    num_test_episodes,
    graph_every,
):
    print(f"Starting: {graph_name}")

    if agents[0].n != 0:
        graph_name += "_" + str(agents[0].n)

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    os.makedirs("./graphs", exist_ok=True)

    num_trials = len(agents)
    num_eval_points = num_episodes // graph_every
    D = np.zeros((num_trials, num_eval_points), dtype=np.float32)

    for trial, agent in enumerate(agents):
        eval_idx = 0
        episodes_done = 0

        while episodes_done < num_episodes:
            # Train for graph_every episodes
            agent.run(env, max_steps, graph_every, train=True)
            episodes_done += graph_every

            # Evaluate frozen policy
            eval_rewards = agent.run(env, max_steps, num_test_episodes, train=False)
            D[trial, eval_idx] = float(np.mean(eval_rewards))
            eval_idx += 1

    average_total_rewards = np.mean(D, axis=0)
    min_total_rewards = np.min(D, axis=0)
    max_total_rewards = np.max(D, axis=0)
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Total Reward")
    fig.savefig(f"./graphs/{graph_name}.png")
    plt.close(fig)
    print(f"Finished: {graph_name}")


def parse_args():
    mode_choices = ["REINFORCE", "REINFORCE_WITH_BASELINE", "A2C"]

    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument(
        "--mode",
        type=str,
        default="REINFORCE",
        choices=mode_choices,
        help="Mode to run the agent in",
    )
    parser.add_argument("--n", type=int, default=0, help="The n to use for n step A2C")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over for graph",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3500, help="Number of episodes to train for"
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=20,
        help="Number of episodes to test for every eval step",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps in the environment",
    )
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--graph_every", type=int, default=100, help="Graph every x episodes"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a single environment to be reused (Gymnasium API)
    env = gym.make(args.env_name)

    agents = []
    for seed in range(args.num_runs):
        torch.manual_seed(seed)
        np.random.seed(seed)

        dummy_env = gym.make(args.env_name)
        state_size = dummy_env.observation_space.shape[0]
        action_size = dummy_env.action_space.n
        dummy_env.close()

        agent = PolicyGradient(
            state_size=state_size,
            action_size=action_size,
            lr_actor=1e-3,
            lr_critic=1e-3,
            mode=args.mode,
            n=args.n,
            gamma=0.99,
            device=device,
        )
        agents.append(agent)

    graph_name = args.mode
    graph_agents(
        graph_name=graph_name,
        agents=agents,
        env=env,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes,
        num_test_episodes=args.num_test_episodes,
        graph_every=args.graph_every,
    )

    env.close()
    # END STUDENT SOLUTION


if "__main__" == __name__:
    main()