import gymnasium as gym
import os
import numpy as np
import torch
from torch import nn

from tqdm import tqdm
import pickle

from diffusion_policy_transformer import PolicyDiffusionTransformer
from PIL import Image
from diffusers import DDPMScheduler, DDIMScheduler
import time
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

class TrainDiffusionPolicy:

    def __init__(
        self,
        env,
        model, 
        optimizer, 
        states_array, 
        actions_array, 
        device="cpu", 
        num_train_diffusion_timesteps=30,
        max_trajectory_length=1600,
    ):
        """
        Initializes the TrainDiffusionPolicy class. Creates necessary data structures and normalizes states AND actions.

        Args:
            env (gym.Env): The environment that the model is trained on.
            model (PolicyDiffusionTransformer): the model to train
            optimizer (torch.optim.Optimizer): the optimizer to use for training the model
            states_array (np.ndarray): the states to train on
            actions_array (np.ndarray): the actions to train on
            device (str): the device to use for training
            num_train_diffusion_timesteps (int): the number of diffusion timesteps to use for training
        """
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.states = states_array
        self.actions = actions_array

        self.action_dimension = self.actions.shape[-1]
        self.state_dimension = self.states.shape[-1]

        # clip all actions to be between -1 and 1, as this is the range that the environment expects
        self.clip_sample_range = 1
        self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        self.trajectory_lengths = [sum(1 for s in self.states[i] if np.sum(s) != 0) for i in range(len(self.states))]
        self.max_trajectory_length = max_trajectory_length

        model.set_device(self.device)

        # normalize states and actions
        all_states = np.concatenate([self.states[i, 0:self.trajectory_lengths[i]] for i in range(len(self.states))], axis=0)
        all_actions = np.concatenate([self.actions[i, 0:self.trajectory_lengths[i]] for i in range(len(self.actions))], axis=0)

        self.states_mean = np.mean(all_states, axis=(0))
        self.states_std = np.std(all_states, axis=(0))
        self.states = (self.states - self.states_mean) / self.states_std

        self.actions_mean = np.mean(all_actions, axis=(0))
        self.actions_std = np.std(all_actions, axis=(0))
        self.actions = (self.actions - self.actions_mean) / self.actions_std

        self.num_train_diffusion_timesteps = num_train_diffusion_timesteps

        # training and inference schedulers for diffusion
        self.training_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log", # variance is different for inference, see paper https://arxiv.org/pdf/2301.10677
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler.alphas_cumprod = self.inference_scheduler.alphas_cumprod.to(self.device)


    def get_inference_timesteps(self):
        """
        gets the timesteps to use for inference
        """
        self.inference_scheduler.set_timesteps(self.num_train_diffusion_timesteps, device=self.device)
        return self.inference_scheduler.timesteps

    def diffusion_sample(
        self,
        previous_states, 
        previous_actions,
        episode_timesteps,
        previous_states_padding_mask=None,
        previous_actions_padding_mask=None,
        actions_padding_mask=None,
        max_action_len=3,
    ):
        """
        perform a single diffusion sample from noise to actions

        Args:
            previous_states (torch.Tensor): the previous states to condition on
            previous_actions (torch.Tensor): the previous actions to condition on
            episode_timesteps (torch.Tensor): the episode timesteps to condition on
            previous_states_padding_mask (torch.Tensor): the padding mask for the previous states
            previous_actions_padding_mask (torch.Tensor): the padding mask for the previous actions
            actions_padding_mask (torch.Tensor): the padding mask for the actions being predicted
            max_action_len (int): the maximum number of actions to predict

        NOTE: remember that you are predicting max_action_len actions, not just one
        """
        # BEGIN STUDENT SOLUTION

        self.model.eval()

        B = previous_states.shape[0]
        act_dim = self.action_dimension

        # Start from pure Gaussian noise for the future actions we want to predict
        xt = torch.randn(B, max_action_len, act_dim, device=self.device)

        # Inference timesteps come in decreasing order (T, …, 1)
        timesteps = self.get_inference_timesteps()  # tensor on self.device

        with torch.no_grad():
            for t in timesteps:
                # (B, 1) noise-level “time” input expected by the model
                t_in = torch.full((B, ), int(t.item()), device=self.device, dtype=torch.long)

                # ε_θ(prev_states, prev_actions, xt, episode_timesteps, t)
                pred_eps = self.model(
                    previous_states=previous_states,                          # (B, k_s, state_dim)
                    previous_actions=previous_actions,                        # (B, k_a, act_dim)
                    noisy_actions=xt,                                         # (B, k',  act_dim)
                    episode_timesteps=episode_timesteps,                      # (B, k_s)
                    noise_timesteps=t_in,                                     # (B, 1)
                    previous_states_mask=previous_states_padding_mask,        # (B, k_s)  True = pad
                    previous_actions_mask=previous_actions_padding_mask,      # (B, k_a)  True = pad
                    actions_padding_mask=actions_padding_mask,                # (B, k')   True = pad
                )

                # One reverse-diffusion step: x_{t-1} ← scheduler.step(ε̂, t, x_t)
                step_out = self.inference_scheduler.step(
                    model_output=pred_eps,
                    timestep=t,
                    sample=xt,
                )
                xt = step_out.prev_sample  # becomes x_{t-1}

        # After the loop, xt is x_0 (denoised, *normalized* actions)
        predicted_actions = xt
        # END STUDENT SOLUTION
        return predicted_actions

    def sample_trajectory(
        self, 
        env, 
        num_actions_to_eval_in_a_row=3, 
        num_previous_states=5,
        num_previous_actions=4, 
        render=False,
    ):
        """
        run a trajectory using the trained model

        Args:
            env (gym.Env): the environment to run the trajectory in
            num_actions_to_eval_in_a_row (int): the number of actions to evaluate in a row
            num_previous_states (int): the number of previous states to condition on
            num_previous_actions (int): the number of previous actions to condition on
            render (bool): whether to save images from environment

        NOTE: use with torch.no_grad(): to speed up inference by not storing gradients
        NOTE: for the first few steps, make sure to add padding to previous states/actions - use False if a state/action should be included, and True if it should be padded
        NOTE: both states and actions should be normalized before being passed to the model, and the model outputs normalized actions that need to be denormalized
        NOTE: refer to the forward function of diffusion_policy_transformer to see how to pass in the inputs (tensor shapes, etc.)
        """
        rewards, rgbs = np.zeros((self.max_trajectory_length,)), []
        # BEGIN STUDENT SOLUTION
        self.model.eval()
        rewards[:] = 0.0
        rgbs = []

        state_dim = self.state_dimension
        act_dim = self.action_dimension

        # Keep recent context (lists); we’ll pad on the RIGHT (end) to match the model’s expectation
        prev_states_buf, prev_actions_buf = [], []

        obs, _ = env.reset()
        if render:
            rgbs.append(env.render())

        with torch.no_grad():
            t_env = 0
            done = False
            truncated = False

            while (not done) and (not truncated) and (t_env < self.max_trajectory_length):

                # Build conditioning windows
                ps = prev_states_buf[-num_previous_states:]
                pa = prev_actions_buf[-num_previous_actions:]

                # Normalize and pad previous states
                ps_norm = [(np.array(s, dtype=np.float32) - self.states_mean) / self.states_std for s in ps]
                pad_ps = num_previous_states - len(ps_norm)
                if pad_ps > 0:
                    ps_norm = ps_norm + [np.zeros(state_dim, dtype=np.float32)] * pad_ps
                ps_mask = [False] * (num_previous_states - pad_ps) + [True] * pad_ps  # False=keep, True=pad

                # Normalize and pad previous actions
                pa_norm = [(np.array(a, dtype=np.float32) - self.actions_mean) / self.actions_std for a in pa]
                pad_pa = num_previous_actions - len(pa_norm)
                if pad_pa > 0:
                    pa_norm = pa_norm + [np.zeros(act_dim, dtype=np.float32)] * pad_pa
                pa_mask = [False] * (num_previous_actions - pad_pa) + [True] * pad_pa

                # Episode timesteps matching previous_states length
                epi_ts = np.arange(max(1, num_previous_states), dtype=np.int64)  # simple positional steps
                epi_ts = epi_ts - 1  # start at 0

                # Make tensors (batch=1)
                prev_states_t = torch.tensor(ps_norm, dtype=torch.float32, device=self.device).unsqueeze(0)          # (1, k_s, state_dim)
                prev_actions_t = torch.tensor(pa_norm, dtype=torch.float32, device=self.device).unsqueeze(0)        # (1, k_a, act_dim)
                ep_ts_t = torch.tensor(epi_ts, dtype=torch.long, device=self.device).unsqueeze(0)                    # (1, k_s)

                ps_mask_t = torch.tensor(ps_mask, dtype=torch.bool, device=self.device).unsqueeze(0)                 # (1, k_s)
                pa_mask_t = torch.tensor(pa_mask, dtype=torch.bool, device=self.device).unsqueeze(0)                 # (1, k_a)
                act_mask_t = torch.zeros((1, num_actions_to_eval_in_a_row), dtype=torch.bool, device=self.device)    # predict all steps

                # Sample k future actions (normalized)
                pred_norm_actions = self.diffusion_sample(
                    previous_states=prev_states_t,
                    previous_actions=prev_actions_t,
                    episode_timesteps=ep_ts_t,
                    previous_states_padding_mask=ps_mask_t,
                    previous_actions_padding_mask=pa_mask_t,
                    actions_padding_mask=act_mask_t,
                    max_action_len=num_actions_to_eval_in_a_row,
                )[0].cpu().numpy()  # (k, act_dim)

                # Denormalize and clip
                pred_actions = pred_norm_actions * self.actions_std + self.actions_mean
                pred_actions = np.clip(pred_actions, -self.clip_sample_range, self.clip_sample_range)

                # Execute up to k actions (stop early if episode ends)
                for i in range(num_actions_to_eval_in_a_row):
                    a = pred_actions[i]
                    next_obs, r, done, truncated, _ = env.step(a)

                    # log
                    rewards[t_env] = r
                    if render:
                        rgbs.append(env.render())

                    # update buffers (store raw obs/action; they’ll be normalized next iter)
                    prev_states_buf.append(obs)
                    prev_actions_buf.append(a)

                    t_env += 1
                    obs = next_obs

                    if done or truncated or (t_env >= self.max_trajectory_length):
                        break
        # END STUDENT SOLUTION
        return rewards, rgbs

    def evaluation(
        self,
        diffusion_policy_iter=None, 
        num_samples=20,
        num_actions_to_eval_in_a_row=3,
    ):
        """
        evaluate the model on the environment

        Args:
            diffusion_policy_iter (Optional[int]): the iteration to load the diffusion policy from
            num_samples (int): the number of samples to evaluate

        NOTE: feel free to change this function when making graphs
        """
        # load model weights:
        if diffusion_policy_iter is None:
            self.model.load_state_dict(torch.load(f"data/diffusion_policy_transformer_models/diffusion_policy.pt", map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(f"data/diffusion_policy_transformer_models/diffusion_policy_iter_{diffusion_policy_iter}.pt", map_location=self.device))

 
        self.model.eval() # turn on eval mode (this turns off dropout, running_mean, etc. that are used in training)

        rewards = np.zeros((num_samples, self.max_trajectory_length))
        os.makedirs("data/diffusion_policy_trajectories", exist_ok=True)
        for sample_trajectory in tqdm(range(num_samples)):
            time1 = time.time()
            reward, _ = self.sample_trajectory(self.env, num_actions_to_eval_in_a_row=num_actions_to_eval_in_a_row)
            time2 = time.time()
            print(f"trajectory {sample_trajectory} took {time2 - time1} seconds")
            rewards[sample_trajectory] = reward
            print(f"rewards from trajectory {sample_trajectory}={reward.sum()}")
        print(f"average reward per trajectory={rewards.sum() / (rewards.shape[0])}")
        print(f"median reward per trajectory={np.median(rewards.sum(axis=1))}")
        print(f"max reward per trajectory={np.max(rewards.sum(axis=1))}")
        print(f"average trajectory length={np.mean(np.array([sum(1 for r in rewards[i] if r != 0) for i in range(len(rewards))]))}")

    def train(
        self, 
        num_training_steps, 
        batch_size=64, 
        print_every=5000, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        training loop that calls training_step

        Args:
            num_training_steps (int): the number of training steps to run
            batch_size (int): the batch size to use
            print_every (int): how often to print the loss
            save_every (int): how often to save the model
            wandb_logging (bool): whether to log to wandb
        """
        model = self.model
        if wandb_logging:
            wandb.init(
                name="diffusion transfomer training",
                group="diffuson transformer",
                project='walker deepRL HW3',
            )

        losses = np.zeros(num_training_steps)
        model.train()
        for training_iter in tqdm(range(num_training_steps)):
            loss = self.training_step(batch_size)
            losses[training_iter] = loss
            if wandb_logging:
                wandb.log({"loss": loss})
            if training_iter % print_every == 0:
                print(f"Training Iteration {training_iter}: loss = {loss}")
            if (training_iter + 1) % save_every == 0:
                # save model in data/diffusion_policy_transformer_models
                os.makedirs("data/diffusion_policy_transformer_models", exist_ok=True)
                torch.save(model.state_dict(), f"data/diffusion_policy_transformer_models/diffusion_policy_iter_{training_iter + 1}.pt")

        os.makedirs("data/diffusion_policy_transformer_models", exist_ok=True)
        torch.save(model.state_dict(), f"data/diffusion_policy_transformer_models/diffusion_policy.pt")
        if wandb_logging:
            wandb.finish()
        else:
            x_axis = np.arange(num_training_steps)
            plt.plot(x_axis, losses)
            plt.xlabel("Training Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss Diffusion Policy")
            plt.savefig("data/diffusion_policy_transformer_models/diffusion_policy_loss.png")
            print(f"final loss={losses[-1]}")

        return losses

    def training_step(self, batch_size):
        """
        Runs a single training step on the model.

        Args:
            batch_size (int): The batch size to use.

        NOTE: actions_padding is a mask that is False for actions to be predicted and True otherwise 
                (for instance, the model predicts 3 actions, but our batch element may contain the 2 final actions in a sequence)
                when calculating the loss, we should only consider the loss for the actions that are not padded
        NOTE: return a loss value that is a plain float (not a tensor), and is on cpu
        """
        # BEGIN STUDENT SOLUTION
        self.model.train()

        # Get a batch: previous states/actions, clean future actions (k′), episode timesteps,
        # and the three padding masks (see get_training_batch provided in this file).
        (
            prev_states,           # (B, k, state_dim)
            prev_actions,          # (B, k-1, act_dim)
            clean_future_actions,  # (B, k', act_dim)
            episode_timesteps,     # (B, k)
            ps_mask,               # (B, k)      False=keep, True=pad
            pa_mask,               # (B, k-1)    False=keep, True=pad
            act_mask               # (B, k')     False=keep, True=pad   (all False in training typically)
        ) = self.get_training_batch(batch_size=batch_size)

        B, Kp, act_dim = clean_future_actions.shape  # Kp = k'

        # Move to device
        prev_states = prev_states.to(self.device)
        prev_actions = prev_actions.to(self.device)
        clean_future_actions = clean_future_actions.to(self.device)
        episode_timesteps = episode_timesteps.to(self.device).long()
        ps_mask = ps_mask.to(self.device)
        a_mask = pa_mask.to(self.device)
        act_mask = act_mask.to(self.device)

        # Sample noise and a diffusion timestep per element
        eps = torch.randn_like(clean_future_actions)                     # ϵ ~ N(0, I)
        t = torch.randint(
            low=1, high=self.num_train_diffusion_timesteps, size=(B,), device=self.device
        ).long()                                                         # one t per item

        # Add noise to clean actions using the TRAINING scheduler (DDPM)
        noisy_actions = self.training_scheduler.add_noise(
            clean_future_actions, noise=eps, timesteps=t
        )                                                                # (B, k', act_dim)

        # Predict noise with the model
        pred_eps = self.model(
            previous_states=prev_states,
            previous_actions=prev_actions,
            noisy_actions=noisy_actions,
            episode_timesteps=episode_timesteps,
            noise_timesteps=t,           # (B,1) as expected by the transformer
            previous_states_mask=ps_mask,
            previous_actions_mask=pa_mask,
            actions_padding_mask=act_mask,
        )

        # MSE(ϵθ(·), ϵ)
        loss = torch.nn.MSELoss()(pred_eps, eps)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())
        # END STUDENT SOLUTION

        return loss

    def get_training_batch(self, batch_size, max_action_len=3, num_previous_states=5, num_previous_actions=4):
        """
        get a training batch for the model
        Args:
            batch_size (int): the batch size to use
            max_action_len (int): the maximum number of actions to predict
            num_previous_states (int): the number of previous states to condition on
            num_previous_actions (int): the number of previous actions to condition on
        """
        assert num_previous_states == num_previous_actions + 1, f"num_previous_states={num_previous_states} must be equal to num_previous_actions + 1={num_previous_actions + 1}"

        # get trajectory lengths, so we can sample each trajectory with probability proportional to its length
        # this is equivalent to sampling uniformly from the set of all environment steps
        batch_indices = np.random.choice(
            np.arange(len(self.trajectory_lengths)),
            size=batch_size,
            replace=True,
            p=np.array(self.trajectory_lengths) / sum(self.trajectory_lengths)
        )

        previous_states_batch, previous_actions_batch, actions_batch, episode_timesteps_batch, previous_states_padding_batch, previous_actions_padding_batch, actions_padding_batch = [], [], [], [], [], [], []
        for i in range(len(batch_indices)):
            # get the start and end index for states to condition on
            end_index_state = np.random.randint(1, self.trajectory_lengths[batch_indices[i]])
            start_index_state = max(0, end_index_state - num_previous_states)

            # get the start and end index for actions to condition on (we predict the action for the final state)
            start_index_previous_actions = start_index_state
            end_index_previous_actions = end_index_state - 1

            # get the start and end index for actions to predict
            start_index_action = end_index_state
            end_index_action = min(self.trajectory_lengths[batch_indices[i]], start_index_action + max_action_len)

            previous_states = self.states[batch_indices[i], start_index_state:end_index_state]
            previous_actions = self.actions[batch_indices[i], start_index_previous_actions:end_index_previous_actions]
            actions = self.actions[batch_indices[i], start_index_action:end_index_action]

            state_dim = previous_states.shape[1]
            action_dim = actions.shape[1]

            state_seq_length = previous_states.shape[0]
            previous_action_seq_length = previous_actions.shape[0]

            # if we have less than the max number of previous states, add some padding (i.e. we're predicting a very early state)
            if state_seq_length < num_previous_states:
                previous_states = np.concatenate([previous_states, np.zeros((num_previous_states - state_seq_length, state_dim))], axis=0)
                previous_actions = np.concatenate([previous_actions, np.zeros((num_previous_actions - previous_action_seq_length, action_dim))], axis=0)
                previous_states_padding_mask = np.concatenate([np.zeros(state_seq_length), np.ones(num_previous_states - state_seq_length)], axis=0)
                previous_actions_padding_mask = np.concatenate([np.zeros(previous_action_seq_length), np.ones(num_previous_actions - previous_action_seq_length)], axis=0)
            else:
                previous_states_padding_mask = np.zeros(num_previous_states)
                previous_actions_padding_mask = np.zeros(num_previous_actions)

            # if we have less than the max number of actions, add some padding (i.e. we're predicting a very early action)
            action_seq_length = actions.shape[0]
            if action_seq_length < max_action_len:
                action_dim = actions.shape[1]
                actions = np.concatenate([actions, np.zeros((max_action_len - action_seq_length, action_dim))], axis=0)
                action_padding_mask = np.concatenate([np.zeros(action_seq_length), np.ones(max_action_len - action_seq_length)], axis=0)

            else:
                action_padding_mask = np.zeros(max_action_len)

            previous_states_batch.append(previous_states)
            previous_actions_batch.append(previous_actions)
            actions_batch.append(actions)
            episode_timesteps_batch.append(np.arange(start_index_state, start_index_state + num_previous_states)) # add extra dummy timesteps in some cases
            previous_states_padding_batch.append(previous_states_padding_mask)
            previous_actions_padding_batch.append(previous_actions_padding_mask)
            actions_padding_batch.append(action_padding_mask)

        previous_states_batch = np.stack(previous_states_batch)
        previous_actions_batch = np.stack(previous_actions_batch)
        actions_batch = np.stack(actions_batch)
        episode_timesteps_batch = np.stack(episode_timesteps_batch)
        previous_states_padding_batch = np.stack(previous_states_padding_batch)
        previous_actions_padding_batch = np.stack(previous_actions_padding_batch)
        actions_padding_batch = np.stack(actions_padding_batch)

        previous_states_batch = torch.from_numpy(previous_states_batch).float().to(self.device)
        previous_actions_batch = torch.from_numpy(previous_actions_batch).float().to(self.device)
        actions_batch = torch.from_numpy(actions_batch).float().to(self.device)
        previous_states_padding_batch = torch.from_numpy(previous_states_padding_batch).bool().to(self.device)
        previous_actions_padding_batch = torch.from_numpy(previous_actions_padding_batch).bool().to(self.device)
        actions_padding_batch = torch.from_numpy(actions_padding_batch).bool().to(self.device)
        episode_timesteps_batch = torch.from_numpy(episode_timesteps_batch).long().to(self.device)

        return previous_states_batch, previous_actions_batch, actions_batch, episode_timesteps_batch, previous_states_padding_batch, previous_actions_padding_batch, actions_padding_batch

def run_training():
    """
    Creates the environment, model, and optimizer, loads the data, and trains/evaluates the model using the TrainDiffusionPolicy class.
    """

    env = gym.make('BipedalWalker-v3',render_mode="rgb_array") # , render_mode="rgb_array"
    with open(f"data/states_BC.pkl", "rb") as f:
        states = pickle.load(f)
    with open(f"data/actions_BC.pkl", "rb") as f:
        actions = pickle.load(f)
    # BEGIN STUDENT SOLUTION
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

    # Load expert dataset (states/actions) per the handout
    with open("data/states_BC.pkl", "rb") as f:
        states = pickle.load(f)
    with open("data/actions_BC.pkl", "rb") as f:
        actions = pickle.load(f)

    # Build diffusion transformer policy
    model = PolicyDiffusionTransformer(
        state_dim=states.shape[-1],
        act_dim=actions.shape[-1],
        num_transformer_layers=6,
        hidden_size=128,
        n_transformer_heads=1,
        target="diffusion_policy",
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

    # Wrap in trainer (this class normalizes states/actions internally)
    trainer = TrainDiffusionPolicy(
        env=env,
        model=model,
        optimizer=optimizer,
        states_array=states,
        actions_array=actions,
        device=device,
        num_train_diffusion_timesteps=30,  # per spec
        max_trajectory_length=1600,
    )

    # Train for 50k steps with batch=256, save at end
    losses = []
    num_steps = 50_000
    batch_size = 256
    t0 = time.time()

    for step in range(1, num_steps + 1):
        loss = trainer.training_step(batch_size=batch_size)
        losses.append(loss)

        if step % 1000 == 0:
            print(f"[Diffusion] step {step}/{num_steps} | loss {loss:.4f}")

    # Save model + loss plot
    torch.save(model.state_dict(), "diffusion_policy.pt")
    plt.figure()
    plt.plot(losses)
    plt.title("Diffusion Policy Training Loss")
    plt.xlabel("Step"); plt.ylabel("MSE to noise")
    plt.grid(True)
    plt.savefig("diffusion_training_loss.png")

    print(f"Finished 50k steps in {time.time()-t0:.1f}s; final loss={losses[-1]:.4f}")
    # END STUDENT SOLUTION
    trainer.evaluation(num_samples=30)
    traj_reward = 0
    while traj_reward < 240:
        rewards, rgbs = trainer.run_trajectory(trainer.env, num_actions_to_eval_in_a_row=3, render=True)
        traj_reward = rewards.sum()
        print(f"got trajectory with reward {traj_reward}")
    imageio.mimsave(f'gifs_diffusion.gif', rgbs, fps=33)

if __name__ == "__main__":
    run_training()
