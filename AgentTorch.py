# PyTorch model and training necessities
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, ExponentialLR
from torch.distributions import Categorical

# Image display
import pygame
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import os
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        """
        Early stops the training if validation reward doesn't improve after a given patience.

        Args:
            patience (int): How long to wait after last time validation reward improved.
                            Default: 7
            delta (float):  Minimum change (in %) in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_reward):
        score = val_reward

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score * (1 + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
    

class PPOAgent:
    DEFAULTS = {
        'actor_lr': 0.005,
        'critic_lr': 0.005,
        'decay_method': 'exponential',
        'exponential_factor': 0.95,
        'value_loss_factor': 0.5,
        'entropy': 0.0001, 
        'gamma': 0.99,
        'GAE_lambda': 0.95,
        'clipping_epsilon': 0.2,  # Proposed value in Schulman 2017
        'l1_factor': 0.00001,
        'l2_factor': 0.00001,
        'T': 512,
        'minibatch_size': 64,
        'epochs': 10,
        'updates': 100,
        'val_episodes': 10,
        'updates_per_val': 1,
        'target_kl': 0.02,
        'adv_std': True,   
        'early_stopping_patience': 15,
        'early_stopping_delta': 0
    }
    def __init__(self, actor_model, critic_model, log_dir='runs\\ppo_experiment', purge_step=None, smoothing_factor=0.2, **kwargs):
        self.params = {**self.DEFAULTS, **kwargs}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_model = actor_model.to(self.device)
        self.critic_model = critic_model.to(self.device)
        print(next(self.actor_model.parameters()).device)

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.params['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.params['critic_lr'])

        if self.params['decay_method'] is not None:
            if self.params['decay_method'] == 'plateau':
                self.actor_scheduler = ReduceLROnPlateau(optimizer=self.actor_optimizer, mode='max', factor=self.params['plateau_factor'], patience=self.params['plateau_patience'], verbose=False)
                self.critic_scheduler = ReduceLROnPlateau(optimizer=self.critic_optimizer, mode='max', factor=self.params['plateau_factor'], patience=self.params['plateau_patience'], verbose=False)
            elif self.params['decay_method'] == 'linear':
                self.actor_scheduler = LinearLR(optimizer=self.actor_optimizer, start_factor=1.0, end_factor=self.params['linear_end_factor'], total_iters=self.params['updates']//self.params['updates_per_val'], verbose=False)
                self.critic_scheduler = LinearLR(optimizer=self.critic_optimizer, start_factor=1.0, end_factor=self.params['linear_end_factor'], total_iters=self.params['updates']//self.params['updates_per_val'], verbose=False)
            elif self.params['decay_method'] == 'exponential':
                self.actor_scheduler = ExponentialLR(self.actor_optimizer, gamma=self.params['exponential_factor'])
                self.critic_scheduler = ExponentialLR(self.critic_optimizer, gamma=self.params['exponential_factor'])
             
        self.val_reward_ema = None  # Initialize EMA for validation reward
        self.ema_alpha = smoothing_factor  # Smoothing factor for EMA, can be adjusted

        self.writer = SummaryWriter(log_dir, purge_step=purge_step)

        self.update = 1
        self.best_val_reward = -np.inf  # Use negative infinity to ensure any reward is better
        self.early_stopping = EarlyStopping(patience=self.params['early_stopping_patience'], delta=self.params['early_stopping_delta'])


    def _compute_gae_and_returns(self, rewards, values, next_value, dones):
        """
        Computes the Generalized Advantage Estimation (GAE) and the returns for each time step using PyTorch.

        Parameters:
        - next_value: The value estimate of the next state (V(s_{T+1})), as a PyTorch tensor.
        - rewards: PyTorch tensor of shape [T] containing rewards received at each time step.
        - dones: PyTorch tensor of shape [T] indicating whether each time step is terminal (1 for terminal states, 0 otherwise).
        - values: PyTorch tensor of shape [T] containing the value function estimates V(s_t) for each time step.

        Returns:
        - A tuple of two elements:
            - GAEs: A PyTorch tensor of shape [T] containing the GAE for each time step.
            - Returns: A PyTorch tensor of shape [T] containing the target values for each time step.
        """
        values = torch.cat((values, next_value.squeeze(0)), dim=0)
        gae = 0
        returns = []
        advantages = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.params['gamma'] * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.params['gamma'] * self.params['GAE_lambda'] * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)

        # Convert the lists to PyTorch tensors
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        return advantages, returns


    def _actor_loss(self, actions, probs, old_probs, advantages):
        """
        Computes the PPO actor loss.

        Parameters:
        - actions: The actions taken by the policy.
        - probs: The probabilities from the current policy.
        - old_probs: The probabilities from the old policy (before the update).
        - advantages: The advantage estimates for the actions taken.

        Returns:
        - The computed PPO actor loss.
        """
        # Create pytorch categorical distribution parameterized by the probs
        dist = Categorical(probs)
        old_dist = Categorical(old_probs)

        # Calculate the ratio of new to old probabilities
        log_pi = dist.log_prob(actions.squeeze())
        old_log_pi = old_dist.log_prob(actions.squeeze())
        ratios = torch.exp(log_pi - old_log_pi)  # Without tiny epsilon?
        # print(ratios)  # If the first is not 1, there is a bug
        
        # Calculate the clipped surrogate objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.params['clipping_epsilon'], 1 + self.params['clipping_epsilon']) * advantages
        policy_loss = - torch.min(surr1, surr2).mean()

        # Calculate the entropy bonus
        entropy_bonus = dist.entropy().mean()  # torch.sum(probs * torch.log(probs), axis=1)

        # Combine the policy loss with the entropy bonus
        total_loss = policy_loss - self.params['entropy'] * entropy_bonus  # negative entropy because the optimizer minimizes the loss by default

        # Compute approximate KL divergence
        with torch.no_grad():  # Ensure no gradients are computed for KL divergence calculation
            self.kl_div = F.kl_div(old_log_pi, log_pi, reduction='batchmean', log_target=True)
        
        return total_loss, policy_loss, entropy_bonus
    

    def _critic_loss(self, returns, values):
        """
        Computes the critic loss for the PPO algorithm.

        The critic loss is calculated using the Huber loss function, which is a combination of the mean squared error loss and mean absolute error loss. 
        It behaves like the mean squared error when the error is small, but like the mean absolute error when the error is large.
        This makes it less sensitive to outliers than the squared error loss.

        Parameters:
        - returns: The discounted returns that the agent expects to receive from each state-action pair. These are the target values that the critic is trying to predict.
        - values: The value function predictions made by the critic for each state. These are the predictions that we are trying to improve.

        Returns:
        - The computed Huber loss scaled by the 'value_loss_factor'

        """
        critic_loss = self.params['value_loss_factor'] * F.huber_loss(returns, values.squeeze())  # https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic?hl=en#3_the_actor-critic_loss

        return critic_loss
    

    def _learn_batch(self, observations, advantages, returns, actions, old_probs):
        
        # Zero the gradients before the backward pass
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Forward pass to compute the actor and critic outputs
        actor_output = self.actor_model(observations)  # Current probs
        critic_output = self.critic_model(observations)

        # Compute the losses
        actor_loss, policy_loss, entropy_bonus = self._actor_loss(actions, actor_output, old_probs, advantages)
        critic_loss = self._critic_loss(returns, critic_output)

        # Add L1 and L2 regularization
        l1_reg = sum(param.abs().sum() for param in self.actor_model.parameters())
        l2_reg = sum(param.pow(2.0).sum() for param in self.actor_model.parameters())
        reg_actor_loss = actor_loss + self.params['l1_factor'] * l1_reg + self.params['l2_factor'] * l2_reg

        # Backward pass to compute the gradients
        reg_actor_loss.backward()
        critic_loss.backward()

        # Update the weights
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return reg_actor_loss.item(), actor_loss.item(), policy_loss.item(), entropy_bonus.item(), critic_loss.item()
    

    def validation(self, env, images, fps=30, episodes=10, plot=True, verbose=True, global_step=None):
        # Switch to validation mode
        self.actor_model.eval()
        self.critic_model.eval()

        test_rewards = []
        ep_lengths = []

        if plot:
            clock = pygame.time.Clock()
            width, height = images[0][0].get_width(), images[0][0].get_height()
            window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("DRL Validation")

        for i in range(episodes):
            ep_reward = 0
            ep_length = 0
            observation = env.reset()
            done = False
            while not done:
                if plot:
                    clock.tick(fps)
                    # Draw car and circuit
                    env.draw(window, images)

                # Convert observation to PyTorch tensor and add batch dimension
                observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Get action from the actor model
                with torch.no_grad():
                    action = self.actor_model.get_action(observation_tensor)
                
                # Convert action to numpy array if necessary
                observation, reward, done = env.step(action.item(), verbose=verbose)
                ep_reward += reward
                ep_length += 1
            
            test_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if verbose: print(f"Episode {i+1} completed.")
        
        if plot: 
            pygame.quit()

        # Mean episode duration and mean episode reward
        avg_val_ep_reward = np.mean(test_rewards)
        avg_val_ep_lengths = np.mean(ep_lengths)

        # Update EMA of validation reward
        if self.val_reward_ema is None:
            self.val_reward_ema = avg_val_ep_reward
        else:
            self.val_reward_ema = self.ema_alpha * avg_val_ep_reward + (1 - self.ema_alpha) * self.val_reward_ema

        if global_step is not None:
            self.writer.add_scalar('Reward/Mean_val_reward', avg_val_ep_reward, global_step)
            self.writer.add_scalar('Reward/Smoothed_val_reward', self.val_reward_ema, global_step)
            self.writer.add_scalar('Duration/Mean_val_ep_duration', avg_val_ep_lengths, global_step)

        if verbose: 
            print(f"Validation mean reward = {avg_val_ep_reward:.3f}")
            print(f"Mean episode duration = {avg_val_ep_lengths}")

        return avg_val_ep_reward
    

    def train(self, train_env, val_env, val_images, val_fps=30, val_plot=False, val_verbose=False, save_path=None, updates_per_flush=20):
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # Tensors to store each batch of data
        actions = torch.zeros(size=(self.params['T'],), dtype=torch.int32).to(self.device)
        probs = torch.zeros(size=(self.params['T'], self.params['output_size']), dtype=torch.float32).to(self.device)
        rewards, values, dones = torch.zeros(size=(3, self.params['T']), dtype=torch.float32).to(self.device)
        observations = torch.zeros(size=(self.params['T'], self.params['input_size']), dtype=torch.float32).to(self.device)

        # Validation variables
        val_rewards = []

        # Training loop (we log the average reward of the episodes for each update)
        update_rewards = []
        next_observation = train_env.reset()

        for update in range(self.update, self.update+self.params['updates'], 1):
            # if target_completed: break  # When convergence condition are satisfied the training is finished

            # Switch to validation mode
            self.actor_model.eval()
            self.critic_model.eval()

            for step in range(self.params['T']):
                # Store values
                observations[step] = torch.tensor(next_observation, dtype=torch.float32)
                with torch.no_grad():
                    probs[step] = self.actor_model(torch.unsqueeze(observations[step], 0))  # Add batch dimensions
                    actions[step] = self.actor_model.get_action(torch.unsqueeze(observations[step], 0))  
                    values[step] = self.critic_model(torch.unsqueeze(observations[step], 0))
                next_observation, rewards[step], dones[step] = train_env.step(actions[step].item())

                if dones[step]:
                    next_observation = train_env.reset()

            # Calculation of the average reward per episode of this update    
            with torch.no_grad():
                avg_train_ep_reward, avg_train_ep_length = self._get_avg_reward(rewards, dones)
                update_rewards.append(avg_train_ep_reward)
                self.writer.add_scalar('Reward/Mean_train_reward', avg_train_ep_reward, update)
                self.writer.add_scalar('Duration/Mean_train_ep_duration', avg_train_ep_length, update)

            with torch.no_grad():
                # Next predicted value
                next_value = self.critic_model(torch.unsqueeze(torch.tensor(next_observation, dtype=torch.float32, device=self.device), 0))
                # Returns and advantages
                returns, advantages = self._compute_gae_and_returns(rewards, values, next_value, dones)
                returns, advantages = returns.to(self.device), advantages.to(self.device)

            # At this point we already have the tensors (of size T) actions, probs, rewards, values, dones and observations filled in.
            indices = np.arange(self.params['T'])
            # Switch to training mode
            self.actor_model.train()
            self.critic_model.train()
            # Perform epochs full training steps on the collected data
            for epoch in range(self.params['epochs']):
                np.random.shuffle(indices)  # We create the mini-batches randomly in each epoch.
                for start in range(0, self.params['T'], self.params['minibatch_size']):
                    end = start + self.params['minibatch_size']
                    minibatch_indices = indices[start:end]

                    # Standardization of advantages to improve performance
                    mb_advantages = advantages[minibatch_indices]
                    if self.params['adv_std']:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Train one minibatch
                    reg_actor_loss, actor_loss, policy_loss, entropy_bonus, critic_loss = self._learn_batch(observations[minibatch_indices], mb_advantages, returns[minibatch_indices], actions[minibatch_indices], probs[minibatch_indices]) 
                
                # If the KL divergence calculated from the last mini-batch of this epoch exceeds the maximum allowed threshold, 
                # there is an early stopping and no further updates are performed with the T data batch.                
                if self.params['target_kl'] is not None:
                    if self.kl_div > self.params['target_kl']:
                        break
            
            # Log the metrics
            self.writer.add_scalar('Loss/Policy_loss', policy_loss, update)
            self.writer.add_scalar('Loss/Entropy_bonus', entropy_bonus, update)
            self.writer.add_scalar('Loss/Actor_loss', actor_loss, update)
            self.writer.add_scalar('Loss/KL_divergence', self.kl_div, update)
            self.writer.add_scalar('Loss/Regularized_Actor_loss', reg_actor_loss, update)
            self.writer.add_scalar('Loss/Critic_loss', critic_loss, update)

            print(f"Update [{update}/{self.update+self.params['updates']-1}]\nActor Loss: {reg_actor_loss}\nCritic Loss: {critic_loss}\n")

            y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            self.writer.add_scalar('Metric/Explained_variance', explained_var, update)

            # Validation stage
            if update % self.params['updates_per_val'] == 0:
                val_reward = self.validation(val_env, val_images, fps=val_fps, episodes=self.params['val_episodes'], plot=val_plot, verbose=val_verbose, global_step=update)
                val_rewards.append(val_reward)

                # Early stopping with smoothed validation reward
                if self.early_stopping(self.val_reward_ema): break

                if self.params['decay_method'] == 'plateau':
                    # Use EMA smoothed validation reward for learning rate scheduler
                    self.actor_scheduler.step(self.val_reward_ema)
                    self.critic_scheduler.step(self.val_reward_ema)
                elif self.params['decay_method'] in ['linear', 'exponential']:
                    # Update learning rate with linear or exponential decay
                    self.actor_scheduler.step()
                    self.critic_scheduler.step()

                # Log the learning rate for both actor and critic optimizers
                actor_lr = self.actor_optimizer.param_groups[0]['lr']
                critic_lr = self.critic_optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_rate/Actor', actor_lr, update)
                self.writer.add_scalar('Learning_rate/Critic', critic_lr, update)

                # Model saving
                if val_reward > self.best_val_reward:
                    print(f"New best validation reward reached in update [{update}/{self.update+self.params['updates']-1}]\n")
                    if save_path is not None:
                        torch.save({
                            'next_update': update+1,
                            'best_val_reward': val_reward,
                            'actor_state_dict': self.actor_model.state_dict(),
                            'critic_state_dict': self.critic_model.state_dict(),
                            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                        }, os.path.join(save_path, f'checkpoint_{update}_{val_reward:.2f}.pth'))
                        
                    self.best_val_reward = val_reward

                # if self.best_val_reward > hp_convergence_condition:
                #     target_completed = True
            
            if update % updates_per_flush == 0:
                self.writer.flush()

        return update_rewards, val_rewards
    

    def _get_avg_reward(self, rewards, dones):
        # Identify the indices where `dones` is True
        end_indices = (dones == 1).nonzero(as_tuple=True)[0]
        # Sum the values within each segment and compute the average of these sums
        if len(end_indices) >= 2:
            avg_length = torch.mean(end_indices[1:]-end_indices[:-1], dtype=torch.float32).item()
            avg_reward = rewards[end_indices[0]+1 : end_indices[-1]+1].sum().item() / (len(end_indices)-1)
            return avg_reward, avg_length
        # If no complete episode has elapsed during the update, there is no point in averaging anything.
        # The metric would be meaningless
        else:  
            return np.nan, np.nan
    

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_model.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.update = checkpoint['next_update']
        self.best_val_reward = checkpoint['best_val_reward']
    

    def close_writer(self):
        # Closes the SummaryWriter object when the training is finished.
        self.writer.close()

