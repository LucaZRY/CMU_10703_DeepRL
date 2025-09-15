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
        self.buffer = collections.deque(maxlen=memory_size)
        self.batch_size = batch_size
        # END STUDENT SOLUTION
        # pass

    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # states      = torch.tensor(np.array(states), dtype=torch.float32)
        # actions     = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        # rewards     = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        # next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        # dones       = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones
        # END STUDENT SOLUTION
        # pass

    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.buffer.append(transition)
        # END STUDENT SOLUTION
        # pass



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
            # nn.Linear(hidden_layer_size, hidden_layer_size),
            # nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.replay_memory = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)

        self.q = q_net_init().to(self.device)
        self.target_q = q_net_init().to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr_q_net)
        # END STUDENT SOLUTION


    def forward(self, state, new_state):
        # calculate q value and target
        # use the correct network for the target based on self.double_dqn
        # BEGIN STUDENT SOLUTION
        q_values = self.q(state)

        with torch.no_grad():
            if self.double_dqn:
                next_q_values = self.q(new_state)
                next_actions = next_q_values.argmax(dim=1, keepdim=True)

                next_q_target = self.target_q(new_state)
                target_q_values = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                target_q_values = self.target_q(new_state).max(dim=1).values

        return q_values, target_q_values
        # END STUDENT SOLUTION


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        if stochastic and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q(state)
            return q_values.argmax(dim = 1).item()
        # END STUDENT SOLUTION
        # pass



def graph_agents(
    graph_name, mean_undiscounted_returns, test_frequency, max_steps, num_episodes
):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    average_total_rewards = []
    min_total_rewards = []
    max_total_rewards = []
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    num_runs = args.num_runs
    num_episodes = args.num_episodes
    max_steps = args.max_steps
    env_name = args.env_name
    test_frequency = args.test_frequency

    all_runs_returns = []

    for run in range(num_runs):
        env = gym.make(env_name)
        obs, info = env.reset(seed=run)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DeepQNetwork(state_size, action_size, 
                             double_dqn = args.double_dqn,
                             device = device)
        
        test_returns = []
        global_step = 0

        for episode in range(1, num_episodes+1):
            obs, info = env.reset()
            total_reward = 0.0

            for step in range(max_steps):
                action = agent.get_action(obs, stochastic=True)
                step_out = env.step(action)
                if len(step_out) == 5:
                    new_obs, reward, terminated, truncated, info = step_out
                    done = terminated or truncated
                else:
                    new_obs, reward, done, info = step_out

                agent.replay_memory.append((obs, action, reward, new_obs, done))
                obs = new_obs
                total_reward += reward
                global_step += 1

                if len(agent.replay.buffer) >= max(agent.replay.batch_size, agent.burn_in):
                    states, actions, rewards, next_states, dones = agent.replay.sample_batch()
                    states = states.to(device)
                    actions = actions.to(device)            # (B,1)
                    rewards = rewards.to(device)            # (B,1)
                    next_states = next_states.to(device)
                    dones = dones.to(device)                # (B,1)

                    # Q(s,a; θ)
                    q_sa = agent.q(states).gather(1, actions)

                    # targets
                    with torch.no_grad():
                        if agent.double_dqn:
                            next_actions = agent.q(next_states).argmax(dim=1, keepdim=True)
                            next_q = agent.target_q(next_states).gather(1, next_actions)
                        else:
                            next_q = agent.target_q(next_states).max(dim=1, keepdim=True).values
                        targets = rewards + (1.0 - dones) * agent.gamma * next_q

                    loss = F.smooth_l1_loss(q_sa, targets)
                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.q.parameters(), 10.0)
                    agent.optimizer.step()

                    # periodic target update
                    if global_step % agent.target_update == 0:
                        agent.target_q.load_state_dict(agent.q.state_dict())

                if done:
                    break

            if episode % test_frequency == 0:
                eval_returns = []
                for _ in range(5):
                    reset_out = env.reset()
                    test_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    R = 0.0
                    for _ in range(max_steps):
                        test_action = agent.get_action(test_obs, stochastic=False)
                        step_out = env.step(test_action)
                        if len(step_out) == 5:
                            test_obs, r_eval, term, trunc, _ = step_out
                            d_eval = term or trunc
                        else:
                            test_obs, r_eval, d_eval, _ = step_out
                        R += r_eval
                        if d_eval:
                            break
                    eval_returns.append(R)
                mean_eval = float(np.mean(eval_returns))
                test_returns.append(mean_eval)
                print(f'Run: {run+1}, Episode: {episode}, Eval Avg Return: {mean_eval:.1f}')

        env.close()
        all_runs_returns.append(test_returns)

    graph_agents(
        graph_name=f"DQN_{'Double' if args.double_dqn else 'Vanilla'}_{env_name}",
        mean_undiscounted_returns=all_runs_returns,
        test_frequency=test_frequency,
        max_steps=max_steps,
        num_episodes=num_episodes,)
    
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()

