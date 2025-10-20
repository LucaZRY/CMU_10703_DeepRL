import gymnasium as gym
import numpy as np
import torch
from torch import nn
import argparse
import imageio
from modules import PolicyNet
from simple_network import SimpleNet
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os

try:
    import wandb
except ImportError:
    wandb = None

class TrainDaggerBC:

    def __init__(self, env, model, optimizer, states, actions, expert_model=None, device="cpu", mode="DAgger"):
        """
        Initializes the TrainDAgger class. Creates necessary data structures.

        Args:
            env: an OpenAI Gym environment.
            model: the model to be trained.
            expert_model: the expert model that provides the expert actions.
            device: the device to be used for training.
            mode: the mode to be used for training. Either "DAgger" or "BC".

        """
        self.env = env
        self.model = model
        self.expert_model = expert_model
        self.optimizer = optimizer
        self.device = device
        model.set_device(self.device)

        self.mode = mode

        if self.mode == "BC":
            self.states = []
            self.actions = []
            self.timesteps = []
            for trajectory in range(states.shape[0]):
                trajectory_mask = states[trajectory].sum(axis=1) != 0
                num_steps = trajectory_mask.sum()  # Count actual steps
                self.states.append(states[trajectory][trajectory_mask])
                self.actions.append(actions[trajectory][trajectory_mask])
                self.timesteps.append(np.arange(0, num_steps))  # Use num_steps instead of len(trajectory_mask)
            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            self.timesteps = np.concatenate(self.timesteps, axis=0)

            self.clip_sample_range = 1
            self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        else:
            if expert_model is not None:
                self.expert_model = self.expert_model.to(self.device)
            self.states = None
            self.actions = None
            self.timesteps = None

    def generate_trajectory(self, env, policy, render=False):
        """Collects one rollout from the policy in an environment. The environment
        should implement the OpenAI Gym interface. A rollout ends when done=True. The
        number of states and actions should be the same, so you should not include
        the final state when done=True.

        Args:
            env: an OpenAI Gym environment.
            policy: The output of a deep neural network
            render: Whether to store frames from the environment
            Returns:
            states: a list of states visited by the agent.
            actions: a list of actions taken by the agent. Note that these actions should never actually be trained on...
            timesteps: a list of integers, where timesteps[i] is the timestep at which states[i] was visited.
            rewards: list of rewards given by the environment
            rgbs: list of rgb images from the environment for each timestep
        """

        states, old_actions, timesteps, rewards, rgbs = [], [], [], [], []

        done, trunc = False, False
        cur_state, _ = env.reset()  
        if render:
            rgbs.append(env.render())
        t = 0
        while (not done) and (not trunc):
            with torch.no_grad():
                p = policy(torch.from_numpy(cur_state).to(self.device).float().unsqueeze(0), torch.tensor(t).to(self.device).long().unsqueeze(0))
            a = p.cpu().numpy()[0]
            next_state, reward, done, trunc, _ = env.step(a)

            states.append(cur_state)
            old_actions.append(a)
            timesteps.append(t)
            rewards.append(reward)
            if render:
                rgbs.append(env.render())

            t += 1

            cur_state = next_state

        return states, old_actions, timesteps, rewards, rgbs

    def call_expert_policy(self, state):
        """
        Calls the expert policy to get an action.

        Args:
            state: the current state of the environment.
        """
        # takes in a np array state and returns an np array action
        with torch.no_grad():
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            action = self.expert_model.choose_action(state_tensor, deterministic=True).cpu().numpy()
            action = np.clip(action, -1, 1)[0]
        return action

    def update_training_data(self, num_trajectories_per_batch_collection=20):
        """
        Updates the training data by collecting trajectories from the current policy and the expert policy.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        NOTE: you will need to call self.generate_trajectory and self.call_expert_policy in this function.
        NOTE: you should update self.states, self.actions, and self.timesteps in this function.
        """
        # BEGIN STUDENT SOLUTION
        rewards = []
        states = []
        timesteps = []
        for _ in range(num_trajectories_per_batch_collection):
            new_states, _, new_timesteps, new_rewards, _ = self.generate_trajectory(self.env, self.model)
            rewards += new_rewards
            states += new_states
            timesteps += new_timesteps
            
        expert_actions = np.array([self.call_expert_policy(state) for state in states])

        if self.states is None:
            self.states = np.array(states)
            self.timesteps = np.array(timesteps)
            self.actions = expert_actions
        else:
            self.states = np.concatenate((self.states, np.array(states)))
            self.timesteps = np.concatenate((self.timesteps, np.array(timesteps)))
            self.actions = np.concatenate((self.actions, expert_actions))
        # END STUDENT SOLUTION

        return rewards

    def generate_trajectories(self, num_trajectories_per_batch_collection=20):
        """
        Runs inference for a certain number of trajectories. Use for behavior cloning.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.
        
        NOTE: you will need to call self.generate_trajectory in this function.
        """
        # BEGIN STUDENT SOLUTION
        rewards = []
        for _ in range(num_trajectories_per_batch_collection):
            _, _, _, new_rewards, _ = self.generate_trajectory(self.env, self.model)
            rewards += new_rewards
        # END STUDENT SOLUTION

        return rewards

    def train(
        self, 
        num_batch_collection_steps=20, 
        num_training_steps_per_batch_collection=1000, 
        num_trajectories_per_batch_collection=20, 
        batch_size=64, 
        print_every=500, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        Train the model using BC or DAgger

        Args:
            num_batch_collection_steps: the number of times to collecta batch of trajectories from the current policy.
            num_training_steps_per_batch_collection: the number of times to train the model per batch collection.
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy per batch.
            batch_size: the batch size to use for training.
            print_every: how often to print the loss during training.
            save_every: how often to save the model during training.
            wandb_logging: whether to log the training to wandb.

        NOTE: for BC, you will need to call the self.training_step function and self.generate_trajectories function.
        NOTE: for DAgger, you will need to call the self.training_step and self.update_training_data function.
        """

        losses = np.zeros(num_batch_collection_steps * num_training_steps_per_batch_collection)
        self.model.train()
        mean_rewards, median_rewards, max_rewards = [], [], []
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # BEGIN STUDENT SOLUTION
        for i in tqdm(range(num_batch_collection_steps)):

            if self.mode == "DAgger":
                rewards = self.update_training_data(num_trajectories_per_batch_collection)

            for j in tqdm(range(num_training_steps_per_batch_collection)):
                loss = self.training_step(batch_size)

                step = i * num_training_steps_per_batch_collection + j

                if step % print_every == print_every - 1:
                    print(f"Loss: {loss}")
                    if wandb_logging and wandb is not None:
                        wandb.log({"loss": loss, "step": step})
                
                if step % save_every == save_every - 1:
                    torch.save(self.model.state_dict(), f"models/{self.mode}.pt")

                losses[step] = loss

            if self.mode == "BC":
                rewards = self.generate_trajectories(num_trajectories_per_batch_collection)
                
            mean_rewards.append(np.mean(rewards))
            median_rewards.append(np.median(rewards))
            max_rewards.append(np.max(rewards))
            
            if wandb_logging and wandb is not None:
                wandb.log({
                    "mean_reward": mean_rewards[-1],
                    "median_reward": median_rewards[-1],
                    "max_reward": max_rewards[-1],
                    "batch_step": i
                })

        print(f"FINAL LOSS: {losses[-1]}")
        # END STUDENT SOLUTION
        
        x_axis = np.arange(0, len(mean_rewards)) * num_training_steps_per_batch_collection
        plt.figure()
        plt.plot(x_axis, mean_rewards, label="mean rewards")
        plt.plot(x_axis, median_rewards, label="median rewards")
        plt.plot(x_axis, max_rewards, label="max rewards")
        plt.legend()
        plt.savefig(f"{self.mode}_rewards.png")

        plt.figure()
        plt.plot(np.arange(0, len(losses)), losses, label="training loss")
        plt.legend()
        plt.savefig(f"{self.mode}_losses.png")

        return losses

    def training_step(self, batch_size):
        """
        Simple training step implementation

        Args:
            batch_size: the batch size to use for training.
        """
        states, actions, timesteps = self.get_training_batch(batch_size=batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)

        loss_fn = nn.MSELoss()
        self.optimizer.zero_grad()
        predicted_actions = self.model(states, timesteps)
        loss = loss_fn(predicted_actions, actions)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def get_training_batch(self, batch_size=64):
        """
        get a training batch

        Args:
            batch_size: the batch size to use for training.
        """
        # get random states, actions, and timesteps
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = torch.tensor(self.states[indices], device=self.device).float()
        actions = torch.tensor(self.actions[indices], device=self.device).float()
        timesteps = torch.tensor(self.timesteps[indices], device=self.device).long()
            
        return states, actions, timesteps

def run_training(dagger: bool, use_wandb: bool = False):
    """
    Simple Run Training Function
    """
    
    # Initialize wandb if requested
    if use_wandb and wandb is not None:
        wandb.login(key="0197834dbce2af78d289b7cc7509d28b5a6847f5")
        config = {
            "mode": "DAgger" if dagger else "BC",
            "num_batch_collection_steps": 20,
            "num_training_steps_per_batch_collection": 1000,
            "num_trajectories_per_batch_collection": 20,
            "batch_size": 128,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "hidden_dim": 128,
            "max_timesteps": 1600
        }
        run = wandb.init(
            name="DAgger_trial" if dagger else "BC_trial",
            reinit=True,
            project="DRL_hw4",
            config=config
        )

    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    with open(f"/Users/lucas/Desktop/CMU/10703/F25_703_HW4/code/data/states_BC.pkl", "rb") as f:
        states = pickle.load(f)
    with open(f"/Users/lucas/Desktop/CMU/10703/F25_703_HW4/code/data/actions_BC.pkl", "rb") as f:
        actions = pickle.load(f)

    device = "cpu"
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = SimpleNet(state_dim, action_dim, 128, 1600)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    if dagger:
        # Load expert model
        expert_model = PolicyNet(24, 4)
        model_weights = torch.load(f"/Users/lucas/Desktop/CMU/10703/F25_703_HW4/code/data/models/super_expert_PPO_model.pt", map_location=device)
        expert_model.load_state_dict(model_weights["PolicyNet"])
        # BEGIN STUDENT SOLUTION
        trainer = TrainDaggerBC(env, model, optimizer, None, None, expert_model=expert_model, device=device, mode="DAgger")
        trainer.train(20, 1000, 20, 128, wandb_logging=use_wandb)
        # END STUDENT SOLUTION
        traj_reward = 0
        while traj_reward < 260:
            _, _, _, rewards, rgbs = trainer.generate_trajectory(trainer.env, trainer.model, render=True)
            traj_reward = sum(rewards)
            print(f"got trajectory with reward {traj_reward}")
            imageio.mimsave(f'gifs_{trainer.mode}.gif', rgbs, fps=33)
    else:
        # BEGIN STUDENT SOLUTION
        trainer = TrainDaggerBC(env, model, optimizer, states, actions, device=device, mode="BC")
        trainer.train(20, 1000, 20, 128, wandb_logging=use_wandb)
        # END STUDENT SOLUTION
        traj_reward = 1
        while traj_reward > 0:
            _, _, _, rewards, rgbs = trainer.generate_trajectory(trainer.env, trainer.model, render=True)
            traj_reward = sum(rewards)
            print(f"got trajectory with reward {traj_reward}")
            imageio.mimsave(f'gifs_{trainer.mode}.gif', rgbs, fps=33)
    
    if use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()
    run_training(args.dagger, args.wandb)