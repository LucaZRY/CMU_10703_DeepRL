#! python3

import argparse
import collections
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=memory_size)
        # END STUDENT SOLUTION
        pass


    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        batch = random.sample(self.buffer, k=self.batch_size)
        # each transition: (state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(np.array(states, dtype=np.float32))
        actions = torch.as_tensor(np.array(actions, dtype=np.int64))
        rewards = torch.as_tensor(np.array(rewards, dtype=np.float32))
        next_states = torch.as_tensor(np.array(next_states, dtype=np.float32))
        dones = torch.as_tensor(np.array(dones, dtype=np.float32))
        return states, actions, rewards, next_states, dones
        # END STUDENT SOLUTION
        pass


    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.buffer.append(transition)
        # END STUDENT SOLUTION
        pass



class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, double_dqn, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.replay_memory = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)

        self.q_net = q_net_init().to(self.device)
        self.target_net = q_net_init().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q_net)
        self.train_steps = 0
        # END STUDENT SOLUTION


    def forward(self, state, new_state):
        # calculate q value and target
        # use the correct network for the target based on self.double_dqn
        # BEGIN STUDENT SOLUTION
        # state/new_state: shape [B, state_size]
        q_values = self.q_net(state)  # [B, A]
        with torch.no_grad():
            if self.double_dqn:
                # a* = argmax_a Q_online(s', a)
                a_star = torch.argmax(self.q_net(new_state), dim=1)  # [B]
                target_next_all = self.target_net(new_state)         # [B, A]
                target_q_values = target_next_all.gather(1, a_star.unsqueeze(1)).squeeze(1)  # [B]
            else:
                target_next_all = self.target_net(new_state)         # [B, A]
                target_q_values = torch.max(target_next_all, dim=1).values  # [B]
        return q_values, target_q_values
        # END STUDENT SOLUTION


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        if stochastic and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s)  # [1, A]
        return int(torch.argmax(q, dim=1).item())
        # END STUDENT SOLUTION
        pass



def graph_agents(
    graph_name, mean_undiscounted_returns, test_frequency, max_steps, num_episodes
):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    # mean_undiscounted_returns is expected to be a list (num_runs) of lists (num_checkpoints)
    # Build mean curve with min/max shaded band.
    import os
    os.makedirs("./graphs", exist_ok=True)

    D = np.array(mean_undiscounted_returns, dtype=np.float32)  # [num_runs, num_checkpoints]
    average_total_rewards = D.mean(axis=0)
    min_total_rewards = D.min(axis=0)
    max_total_rewards = D.max(axis=0)
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * test_frequency for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument(
        "--test_frequency",
        type=int,
        default=100,
        help="Number of training episodes between test episodes",
    )
    parser.add_argument("--double_dqn", action="store_true", help="Use Double DQN")
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    device = "cpu"
    env_name = args.env_name
    num_runs = args.num_runs
    num_episodes = args.num_episodes
    test_frequency = args.test_frequency
    max_steps = args.max_steps
    double_flag = args.double_dqn

    # storage for per-run evaluation checkpoints
    all_runs_eval = []

    TEST_EPISODES = 20  # per PDF: run 20 independent tests at each checkpoint

    for run in range(num_runs):
        env = gym.make(env_name)
        obs, info = env.reset(seed=random.randint(0, 10_000_000))
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DeepQNetwork(
            state_size, action_size, double_dqn=double_flag,
            lr_q_net=2e-4, gamma=0.99, epsilon=0.05,
            target_update=1000,  # steps (hard update)
            burn_in=10000, replay_buffer_size=50000,
            replay_buffer_batch_size=32, device=device
        ).to(device)

        # burn-in replay with random policy
        while len(agent.replay_memory.buffer) < agent.burn_in:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_memory.append((obs, action, reward, next_obs, done))
            if done:
                obs, info = env.reset()
            else:
                obs = next_obs

        # training loop
        run_eval = []  # list of mean returns at checkpoints
        global_step = 0
        for ep in range(1, num_episodes + 1):
            obs, info = env.reset()
            ep_steps = 0
            done = False
            while not done and ep_steps < max_steps:
                action = agent.get_action(obs, stochastic=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.replay_memory.append((obs, action, reward, next_obs, done))
                obs = next_obs
                ep_steps += 1
                global_step += 1

                # one gradient step once we have burn-in
                if len(agent.replay_memory.buffer) >= agent.burn_in:
                    states, actions, rewards, next_states, dones = agent.replay_memory.sample_batch()
                    # move to device
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    next_states = next_states.to(device)
                    dones = dones.to(device)

                    q_values, target_next = agent.forward(states, next_states)
                    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                    targets = rewards + (1.0 - dones) * agent.gamma * target_next

                    loss = F.mse_loss(q_sa, targets)

                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()

                    if global_step % agent.target_update == 0:
                        agent.target_net.load_state_dict(agent.q_net.state_dict())

            # evaluate every test_frequency episodes
            if ep % test_frequency == 0:
                # greedy evaluation (ε=0)
                eval_returns = []
                for _ in range(TEST_EPISODES):
                    e = gym.make(env_name)
                    s, _ = e.reset(seed=random.randint(0, 10_000_000))
                    total_r = 0.0
                    d = False
                    steps = 0
                    while not d and steps < max_steps:
                        a = agent.get_action(s, stochastic=False)
                        s, r, term, trunc, _ = e.step(a)
                        d = term or trunc
                        total_r += r
                        steps += 1
                    e.close()
                    eval_returns.append(total_r)
                run_eval.append(float(np.mean(eval_returns)))

        env.close()
        all_runs_eval.append(run_eval)

    graph_name = f"{'Double DQN' if double_flag else 'DQN'} on {env_name}"
    graph_agents(graph_name, all_runs_eval, test_frequency, max_steps, num_episodes)
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
