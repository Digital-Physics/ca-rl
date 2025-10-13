#!/usr/bin/env python3
"""
rl_ca.py

Enhanced Cellular Automata RL with PPO (Proximal Policy Optimization)
Integrates PPO algorithm while preserving all original functionality including:
- Multiple reward types (entropy, maxwell_demon, target_practice, pattern)
- Live visualization during training
- Manual and autonomous demos
- Pattern creator
- Supervised pretraining
- Recording capabilities

Author: Enhanced by Claude with PPO integration
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.signal import convolve2d
from tqdm import tqdm
import argparse
import logging
import os
import time
from collections import deque
import json
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Enable interactive mode for drawing during training
plt.ion()

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --- Custom Keras Layer for Toroidal Padding ---
class ToroidalPadding(layers.Layer):
    """
    Custom Keras layer to apply toroidal (wrap-around) padding to a 4D tensor.
    """
    def __init__(self, **kwargs):
        super(ToroidalPadding, self).__init__(**kwargs)

    def call(self, inputs):
        top_row = inputs[:, -1:, :, :]
        bottom_row = inputs[:, :1, :, :]
        vertical_padded = tf.concat([top_row, inputs, bottom_row], axis=1)

        left_col = vertical_padded[:, :, -1:, :]
        right_col = vertical_padded[:, :, :1, :]
        fully_padded = tf.concat([left_col, vertical_padded, right_col], axis=2)

        return fully_padded

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2, input_shape[2] + 2, input_shape[3])


# --- Cellular Automata Environment ---
class CAEnv:
    """
    Represents the Cellular Automata (CA) environment.
    """
    def __init__(self, grid_size=12, initial_density=0.4, rules_name='conway', 
                 reward_type='entropy', target_pattern=None, max_steps=10):
        self.grid_size = grid_size
        self.initial_density = initial_density
        self.rules_name = rules_name
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.current_step = 0

        self.ca_rules = {
            'conway': {'birth': [3], 'survive': [2, 3]},
            'seeds': {'birth': [2], 'survive': []},
            'maze': {'birth': [3], 'survive': [1, 2, 3, 4, 5]}
        }
        self.rules = self.ca_rules[self.rules_name]

        self.actions = ['up', 'down', 'left', 'right', 'do_nothing'] + [f'write_{i:04b}' for i in range(16)]
        self.num_actions = len(self.actions)

        # Precompute convolution kernel for faster CA updates
        self.ca_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

        if self.reward_type == 'pattern':
            if target_pattern is not None:
                self.target_pattern = target_pattern
            else:
                self.target_pattern = None
        else:
            self.target_pattern = None

        self.reset()

    def _apply_ca_rules_fast(self, grid):
        """Fast CA rule application using convolution."""
        neighbor_counts = convolve2d(grid, self.ca_kernel, mode='same', boundary='wrap')
        
        birth_mask = np.isin(neighbor_counts, self.rules['birth']) & (grid == 0)
        survive_mask = np.isin(neighbor_counts, self.rules['survive']) & (grid == 1)
        
        new_grid = np.zeros_like(grid)
        new_grid[birth_mask | survive_mask] = 1
        
        return new_grid

    def reset(self):
        """Resets the environment to an initial state."""
        if self.reward_type == 'target_practice':
            self.ca_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        elif self.reward_type == 'pattern':
            self.ca_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        else:
            self.ca_grid = (np.random.rand(self.grid_size, self.grid_size) < self.initial_density).astype(np.int8)

        self.agent_x = self.grid_size // 2
        self.agent_y = self.grid_size // 2
        self.current_step = 0

        self.has_target = False
        self.target_x, self.target_y = 0, 0
        if self.reward_type == 'target_practice':
            self._spawn_target()

        return self._get_state()

    def _get_state(self):
        """Constructs the state tensor (grid + agent position channels)."""
        agent_pos_channel = np.zeros_like(self.ca_grid, dtype=np.int8)
        
        for i in range(-1, 1):
             for j in range(-1, 1):
                y = (self.agent_y + i) % self.grid_size
                x = (self.agent_x + j) % self.grid_size
                agent_pos_channel[y,x] = 1

        state = np.stack([self.ca_grid, agent_pos_channel], axis=-1)
        return np.expand_dims(state, axis=0).astype(np.float32)

    def step(self, action):
        """Executes one time step in the environment."""
        reward = 0
        current_pattern = None

        # Execute Agent Action
        if 0 <= action <= 3:
            if action == 0: self.agent_y = (self.agent_y - 1 + self.grid_size) % self.grid_size
            elif action == 1: self.agent_y = (self.agent_y + 1) % self.grid_size
            elif action == 2: self.agent_x = (self.agent_x - 1 + self.grid_size) % self.grid_size
            elif action == 3: self.agent_x = (self.agent_x + 1) % self.grid_size
        elif action == 4:
            pass
        elif action >= 5:
            pattern_index = action - 5
            bits = [(pattern_index >> 3) & 1, (pattern_index >> 2) & 1, (pattern_index >> 1) & 1, pattern_index & 1]
            current_pattern = np.array(bits).reshape(2, 2)

        # Update Cellular Automata (using fast method)
        self._update_ca_fast(current_pattern)
        
        # Calculate Reward
        if self.reward_type == 'entropy':
            new_entropy = self._calculate_entropy()
            reward = -new_entropy
        elif self.reward_type == 'maxwell_demon':
            reward = self._calculate_separation_reward()
        elif self.reward_type == 'target_practice':
            reward = -0.1
            if self._check_target_destroyed():
                reward += 100
                self.ca_grid.fill(0)
                self._spawn_target()
        elif self.reward_type == 'pattern':
            # Match reward: 1.0 = perfect match, 0.0 = no match
            match_fraction = np.mean(self.ca_grid == self.target_pattern) if self.target_pattern is not None else 0.0
            reward = float(match_fraction)

        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        return self._get_state(), reward, done, {}

    def _update_ca_fast(self, write_pattern=None):
        """Fast CA update using convolution."""
        self.ca_grid = self._apply_ca_rules_fast(self.ca_grid)

        # Apply write pattern after CA update
        if write_pattern is not None:
            for i in range(2):
                for j in range(2):
                    y = (self.agent_y + i - 1 + self.grid_size) % self.grid_size
                    x = (self.agent_x + j - 1 + self.grid_size) % self.grid_size
                    self.ca_grid[y, x] = write_pattern[i, j]

    def _calculate_entropy(self):
        p = np.mean(self.ca_grid)
        if p == 0 or p == 1:
            return 0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def _calculate_separation_reward(self):
        midpoint = self.grid_size // 2
        left_density = np.mean(self.ca_grid[:, :midpoint])
        right_density = np.mean(self.ca_grid[:, midpoint:])
        return np.abs(left_density - right_density) * 10

    def _calculate_pattern_bce(self):
        """Calculates Binary Cross-Entropy between current grid and target pattern."""
        if self.target_pattern is None:
            return 0.0
        
        current = self.ca_grid.flatten().astype(np.float32)
        target = self.target_pattern.flatten().astype(np.float32)
        
        epsilon = 1e-7
        current = np.clip(current, epsilon, 1 - epsilon)
        
        # BCE formula: -mean(target * log(current) + (1-target) * log(1-current))
        bce = -np.mean(target * np.log(current) + (1 - target) * np.log(1 - current))
        
        return float(bce)

    def _spawn_target(self):
        self.target_x = np.random.randint(0, self.grid_size - 2)
        self.target_y = np.random.randint(0, self.grid_size - 2)
        self.ca_grid[self.target_y:self.target_y+2, self.target_x:self.target_x+2] = 1
        self.has_target = True

    def _check_target_destroyed(self):
        if not self.has_target:
            return False
        
        target_block = self.ca_grid[self.target_y:self.target_y+2, self.target_x:self.target_x+2]
        if np.sum(target_block) < 4:
            self.has_target = False
            return True
        return False

    def save_pattern(self, filename):
        """Save current grid as target pattern."""
        if filename:
            np.save(filename, self.ca_grid)
            print(f"Pattern saved to {filename}")

    def load_pattern(self, filename):
        """Load pattern from file."""
        if os.path.exists(filename):
            self.target_pattern = np.load(filename)
            print(f"Pattern loaded from {filename}")
            return True
        return False


# --- PPO Actor-Critic Agent ---
class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent with enhanced training stability.
    """
    def __init__(self, state_shape, num_actions, learning_rate=0.0001, gamma=0.99, 
                 lam=0.95, clip_eps=0.1, entropy_coef=0.01, vf_coef=0.5):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam  # GAE lambda
        self.clip_eps = clip_eps  # PPO clip epsilon
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        self._build_networks()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=0.5)

    def _build_networks(self):
        input_layer = layers.Input(shape=self.state_shape)
        
        # Shared CNN base
        x = ToroidalPadding()(input_layer)
        x = layers.Conv2D(32, 3, activation='relu', padding='valid')(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)

        # Actor head
        action_logits = layers.Dense(self.num_actions, name='policy_output')(x)
        
        # Critic head
        state_value = layers.Dense(1, name='value_output')(x)

        self.model = keras.Model(inputs=input_layer, outputs=[action_logits, state_value])
    
    def select_action(self, state):
        """Selects an action based on the policy network's output."""
        logits, _ = self.model(state)
        action_probs = tf.nn.softmax(logits)
        action = tf.random.categorical(logits, 1)[0, 0].numpy()
        return action

    def get_action_and_value(self, state):
        """Get action, log probability, and value for PPO training."""
        logits, value = self.model(state)
        probs = tf.nn.softmax(logits)
        action = tf.random.categorical(logits, 1)[0, 0]
        
        # Calculate log probability
        action_onehot = tf.one_hot(action, self.num_actions)
        selected_prob = tf.reduce_sum(probs * action_onehot, axis=1)
        log_prob = tf.math.log(selected_prob + 1e-8)
        
        return action.numpy(), log_prob.numpy()[0], value.numpy()[0, 0]

    def get_action_probs_and_value(self, state):
        """Get action probabilities and state value for visualization."""
        logits, value = self.model(state)
        action_probs = tf.nn.softmax(logits).numpy()[0]
        value = value.numpy()[0, 0]
        return action_probs, value

    def compute_gae(self, rewards, values, dones, last_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = []
        gae = 0.0
        
        values_ext = values + [last_value]
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    @tf.function
    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        """
        Performs PPO update with clipped objective.
        """
        with tf.GradientTape() as tape:
            logits, values = self.model(states)
            values = tf.squeeze(values, axis=-1)
            
            # Policy loss
            probs = tf.nn.softmax(logits)
            actions_onehot = tf.one_hot(actions, self.num_actions)
            selected_probs = tf.reduce_sum(probs * actions_onehot, axis=1)
            new_log_probs = tf.math.log(selected_probs + 1e-8)
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Value loss
            value_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Entropy bonus
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1))
            
            # Total loss
            total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': total_loss
        }


def supervised_pretrain(args):
    """
    Behavior-cloning + value regression pretraining.
    Expects .npz with arrays: states, actions, rewards, next_states.
    """
    print("--- Supervised Pretraining ---")
    print(args)
    print(args.data_file)
    print(os.path.exists(args.data_file))
    print("Absolute path:", os.path.abspath(args.data_file))

    if not args.data_file or not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Supervised data file not found: {args.data_file}")

    data = np.load(args.data_file)
    states = data['states'].astype(np.float32)
    actions = data['actions'].astype(np.int32)
    rewards = data.get('rewards', np.zeros(len(actions), dtype=np.float32))
    next_states = data.get('next_states', None)
    print(f"Loaded {len(states)} samples from {args.data_file}")

    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward, max_steps=1)
    
    # Load pattern if needed
    if args.reward == 'pattern' and hasattr(args, 'pattern_file') and args.pattern_file:
        env.load_pattern(args.pattern_file)
    
    agent = PPOAgent(
        state_shape=(args.grid_size, args.grid_size, 2),
        num_actions=env.num_actions,
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef
    )

    if args.init_weights and os.path.exists(args.init_weights):
        print(f"Loading initial weights from {args.init_weights}")
        agent.model.load_weights(args.init_weights)

    policy_loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    value_loss_fn = tf.keras.losses.MeanSquaredError()

    batch_size = args.batch_size
    gamma = args.gamma
    N = len(states)
    idx = np.arange(N)
    np.random.shuffle(idx)
    states, actions, rewards = states[idx], actions[idx], rewards[idx]
    if next_states is not None:
        next_states = next_states[idx]

    for epoch in range(args.epochs):
        total_loss, total_acc = 0.0, 0.0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            s_batch = states[start:end]
            a_batch = actions[start:end]
            r_batch = rewards[start:end]
            ns_batch = next_states[start:end] if next_states is not None else None

            with tf.GradientTape() as tape:
                logits, values = agent.model(s_batch, training=True)
                p_loss = policy_loss_fn(a_batch, logits)

                if ns_batch is not None:
                    _, next_values = agent.model(ns_batch, training=False)
                    target_values = r_batch + gamma * tf.squeeze(next_values)
                else:
                    target_values = r_batch

                v_loss = value_loss_fn(target_values, tf.squeeze(values))
                probs = tf.nn.softmax(logits)
                logp = tf.nn.log_softmax(logits)
                entropy = -tf.reduce_mean(tf.reduce_sum(probs * logp, axis=-1))
                loss = p_loss + args.value_coef * v_loss - args.entropy_coef * entropy

            grads = tape.gradient(loss, agent.model.trainable_variables)
            agent.optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))

            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
            total_acc += np.mean(preds.numpy() == a_batch)
            total_loss += float(loss)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={total_loss/(N/batch_size):.4f}, acc={total_acc/(N/batch_size):.4f}")

        if (epoch + 1) % args.save_every == 0:
            fname = f"supervised_epoch{epoch+1}.weights.h5"
            agent.model.save_weights(fname)
            print(f"Saved weights to {fname}")

    agent.model.save_weights(args.save_weights)
    print(f"Saved final supervised weights to {args.save_weights}")


def train_agent(args):
    """Main PPO training loop with rollout collection and minibatch updates."""
    print("--- Starting PPO Training ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward, max_steps=args.rollout_steps)
    
    # Load pattern if reward_type is 'pattern'
    if args.reward == 'pattern' and args.pattern_file:
        env.load_pattern(args.pattern_file)

    agent = PPOAgent(
        state_shape=(args.grid_size, args.grid_size, 2),
        num_actions=env.num_actions,
        learning_rate=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        vf_coef=args.vf_coef
    )

    # Load pretrained weights if provided
    if getattr(args, 'pretrained_weights', None):
        if os.path.exists(args.pretrained_weights):
            print(f"Loading pretrained weights from {args.pretrained_weights}")
            agent.model.load_weights(args.pretrained_weights)
        else:
            print(f"Pretrained weights not found at {args.pretrained_weights} (continuing without).")

    # Metrics tracking
    episode_rewards = []
    policy_losses = []
    value_losses = []
    entropies = []
    density_values = []
    avg_state_values = []

    # Setup live plotting if enabled
    if args.live_plot is not None:
        plt.ion()
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
        
        ax_grid = fig.add_subplot(gs[0, 0])
        if env.reward_type == 'pattern' and env.target_pattern is not None:
            ax_target = fig.add_subplot(gs[0, 1])
            ax_probs = fig.add_subplot(gs[0, 2])
            ax_value_display = fig.add_subplot(gs[0, 3])
        else:
            ax_target = None
            ax_probs = fig.add_subplot(gs[0, 1:3])
            ax_value_display = fig.add_subplot(gs[0, 3])
            
        ax_reward = fig.add_subplot(gs[1, 0])
        ax_loss = fig.add_subplot(gs[1, 1])
        ax_entropy_metric = fig.add_subplot(gs[1, 2])
        ax_density = fig.add_subplot(gs[1, 3])
        ax_value_avg = fig.add_subplot(gs[2, 0])
        
        fig.suptitle('PPO Training Progress (Live)', fontsize=16, fontweight='bold')
        
        grid_img = ax_grid.imshow(env.ca_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                    facecolor='none', edgecolor='cyan', linewidth=2)
        ax_grid.add_patch(agent_patch)
        title_text = ax_grid.set_title("Episode 0 | Step 0", fontweight='bold')
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])

        if ax_target is not None:
            ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
            ax_target.set_title('Target Pattern', fontweight='bold')
            ax_target.set_xticks([])
            ax_target.set_yticks([])

        action_labels = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
        bars = ax_probs.bar(range(env.num_actions), np.zeros(env.num_actions), 
                            color='steelblue', alpha=0.7)
        ax_probs.set_ylim([0, 1])
        ax_probs.set_xticks(range(env.num_actions))
        ax_probs.set_xticklabels(action_labels, fontsize=8)
        ax_probs.set_ylabel('Probability')
        ax_probs.set_title('Action Distribution (Current State)', fontweight='bold')
        ax_probs.grid(axis='y', alpha=0.3)
        
        value_text = ax_value_display.text(0.5, 0.5, 'V(s) = 0.000',
                                   ha='center', va='center', fontsize=18,
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax_value_display.set_xlim([0, 1])
        ax_value_display.set_ylim([0, 1])
        ax_value_display.set_title('State Value (Critic)', fontweight='bold')
        ax_value_display.axis('off')
        
        lines = {
            'reward': ax_reward.plot([], [], 'b-', linewidth=2)[0],
            'policy_loss': ax_loss.plot([], [], 'r-', label='Policy', linewidth=2)[0],
            'value_loss': ax_loss.plot([], [], 'g-', label='Value', linewidth=2)[0],
            'entropy': ax_entropy_metric.plot([], [], 'orange', linewidth=2)[0],
            'density': ax_density.plot([], [], 'darkgreen', linewidth=2)[0],
            'avg_value': ax_value_avg.plot([], [], 'dodgerblue', linewidth=2)[0],
        }
        
        ax_reward.set_title('Episode Reward', fontweight='bold')
        ax_reward.grid(True, alpha=0.3)
        ax_loss.set_title('Losses', fontweight='bold')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        ax_entropy_metric.set_title('Policy Entropy', fontweight='bold')
        ax_entropy_metric.grid(True, alpha=0.3)
        ax_density.set_title('Grid Density', fontweight='bold')
        ax_density.grid(True, alpha=0.3)
        ax_value_avg.set_title('Average State Value', fontweight='bold')
        ax_value_avg.grid(True, alpha=0.3)
        plt.pause(0.1)

    # PPO Training loop
    total_timesteps = 0
    for episode in range(args.episodes):
        # Collect rollout
        states_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        rewards_buffer = []
        values_buffer = []
        dones_buffer = []
        
        state = env.reset()
        episode_reward = 0.0
        episode_density = 0.0
        episode_value_sum = 0.0
        
        pbar = tqdm(range(args.rollout_steps), desc=f"Episode {episode + 1}/{args.episodes}")
        for step in pbar:
            action, log_prob, value = agent.get_action_and_value(state)
            next_state, reward, done, _ = env.step(action)

            # print(reward, value, done)
            
            states_buffer.append(state[0])  # Remove batch dimension
            actions_buffer.append(action)
            log_probs_buffer.append(log_prob)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            dones_buffer.append(done)
            
            episode_reward += reward
            episode_density += np.mean(env.ca_grid)
            episode_value_sum += value
            
            state = next_state
            total_timesteps += 1
            
            pbar.set_postfix({'Reward': f'{episode_reward:.2f}', 'Value': f'{value:.2f}'})
            
            # Live visualization at step level if live_plot == 0
            if args.live_plot == 0:
                grid_img.set_data(env.ca_grid)
                agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
                action_probs, current_value_viz = agent.get_action_probs_and_value(state)
                for bar, prob in zip(bars, action_probs):
                    bar.set_height(prob)
                value_text.set_text(f'V(s) = {current_value_viz:.3f}')
                title_text.set_text(f"Episode {episode + 1} | Step {step + 1} | Reward: {episode_reward:.3f}")
                fig.canvas.draw()
                plt.pause(0.01)
        
        # Get last value for GAE
        _, _, last_value = agent.get_action_and_value(state)
        
        # Compute GAE
        advantages, returns = agent.compute_gae(
            rewards_buffer, values_buffer, dones_buffer, last_value
        )
        
        # Convert to numpy arrays
        states_array = np.array(states_buffer, dtype=np.float32)
        actions_array = np.array(actions_buffer, dtype=np.int32)
        old_log_probs_array = np.array(log_probs_buffer, dtype=np.float32)
        returns_array = np.array(returns, dtype=np.float32)
        advantages_array = np.array(advantages, dtype=np.float32)
        
        # Normalize advantages
        advantages_array = (advantages_array - advantages_array.mean()) / (advantages_array.std() + 1e-8)
        
        # PPO update with minibatches
        num_samples = len(states_array)
        indices = np.arange(num_samples)
        
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        num_updates = 0
        
        for epoch in range(args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, args.batch_size):
                end = min(start + args.batch_size, num_samples)
                batch_indices = indices[start:end]
                
                metrics = agent.ppo_update(
                    states_array[batch_indices],
                    actions_array[batch_indices],
                    old_log_probs_array[batch_indices],
                    returns_array[batch_indices],
                    advantages_array[batch_indices]
                )
                
                epoch_policy_loss += float(metrics['policy_loss'])
                epoch_value_loss += float(metrics['value_loss'])
                epoch_entropy += float(metrics['entropy'])
                num_updates += 1
        
        # Average metrics over updates
        avg_policy_loss = epoch_policy_loss / num_updates
        avg_value_loss = epoch_value_loss / num_updates
        avg_entropy = epoch_entropy / num_updates
        
        # Track episode metrics
        episode_rewards.append(episode_reward)
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)
        entropies.append(avg_entropy)
        density_values.append(episode_density / args.rollout_steps)
        avg_state_values.append(episode_value_sum / args.rollout_steps)
        
        # Update visualizations
        if args.live_plot is not None:
            update_this_episode = False
            if args.live_plot == 0:
                # Already updated during steps, just update metrics graphs
                update_this_episode = True
            elif (episode + 1) % args.live_plot == 0:
                # Update everything including final state view
                update_this_episode = True

            if update_this_episode:
                episodes_range = range(1, episode + 2)
                
                # Update the agent view to show the final state (only if not live_plot==0, since it's already updated)
                if args.live_plot != 0:
                    grid_img.set_data(env.ca_grid)
                    agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
                    action_probs, current_value_viz = agent.get_action_probs_and_value(state)
                    for bar, prob in zip(bars, action_probs):
                        bar.set_height(prob)
                    value_text.set_text(f'V(s) = {current_value_viz:.3f}')

                # Update all metric graphs
                lines['reward'].set_data(episodes_range, episode_rewards)
                lines['policy_loss'].set_data(episodes_range, policy_losses)
                lines['value_loss'].set_data(episodes_range, value_losses)
                lines['entropy'].set_data(episodes_range, entropies)
                lines['density'].set_data(episodes_range, density_values)
                lines['avg_value'].set_data(episodes_range, avg_state_values)

                # Rescale all graph axes
                for ax in [ax_reward, ax_loss, ax_entropy_metric, ax_density, ax_value_avg]:
                    ax.relim()
                    ax.autoscale_view()

                fig.canvas.draw()
                plt.pause(0.01)

        # Save weights periodically
        if (episode + 1) % 25 == 0:
            agent.model.save_weights(f'ca_agent_weights_{episode+1}.weights.h5')
            print(f"\nSaved weights at episode {episode+1}")

    # Save final weights
    agent.model.save_weights('ca_agent_weights_final.weights.h5')
    print("\n--- Training Complete ---")
    
    # Save metrics (convert numpy types to Python types for JSON)
    metrics_data = {
        'episode_rewards': [float(x) for x in episode_rewards],
        'policy_losses': [float(x) for x in policy_losses],
        'value_losses': [float(x) for x in value_losses],
        'entropies': [float(x) for x in entropies],
        'density_values': [float(x) for x in density_values],
        'avg_state_values': [float(x) for x in avg_state_values]
    }
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_data, f)
    print("Training metrics saved to training_metrics.json")

    if args.live_plot is not None:
        plt.ioff()
    
    # Create final summary plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(episode_rewards, linewidth=2)
    plt.title('Episode Reward', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.plot(policy_losses, label='Policy Loss', linewidth=2)
    plt.plot(value_losses, label='Value Loss', linewidth=2)
    plt.title('Training Losses', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 3)
    plt.plot(entropies, linewidth=2, color='orange')
    plt.title('Policy Entropy', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    plt.plot(density_values, linewidth=2, color='darkgreen')
    plt.title('Average Grid Density', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 5)
    plt.plot(avg_state_values, linewidth=2, color='dodgerblue')
    plt.title('Average State Value', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppo_training_results.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("Final plot saved to ppo_training_results.png")


def interactive_pattern_creator(args):
    """Interactive pattern creation tool."""
    print("--- Interactive Pattern Creator ---")
    print("Click cells to toggle them on/off")
    print("Press 's' to save pattern")
    print("Press 'c' to clear grid")
    print("Press 'q' to quit")
    
    grid = np.zeros((args.grid_size, args.grid_size), dtype=np.int8)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Click to toggle cells | s=save | c=clear | q=quit')
    ax.set_xticks(range(args.grid_size))
    ax.set_yticks(range(args.grid_size))
    ax.grid(True, alpha=0.3)
    
    def on_click(event):
        if event.inaxes == ax and event.button == 1:
            x = int(np.round(event.xdata))
            y = int(np.round(event.ydata))
            if 0 <= x < args.grid_size and 0 <= y < args.grid_size:
                grid[y, x] = 1 - grid[y, x]
                img.set_data(grid)
                fig.canvas.draw_idle()
    
    def on_key(event):
        if event.key == 's':
            filename = f'custom_pattern_{args.grid_size}x{args.grid_size}.npy'
            np.save(filename, grid)
            print(f"\nPattern saved to {filename}")
            print(f"Density: {np.mean(grid):.3f}, Live cells: {np.sum(grid)}")
        elif event.key == 'c':
            grid.fill(0)
            img.set_data(grid)
            fig.canvas.draw_idle()
            print("\nGrid cleared")
        elif event.key == 'q':
            plt.close()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show(block=True)


def run_demo_auto(args):
    """Agent-controlled autonomous demo with visualization."""
    import matplotlib as mpl
    mpl.rcParams['keymap.fullscreen'] = []

    print("--- Running Autonomous Demo ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward, max_steps=args.steps)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)

    if not getattr(args, 'weights', None) or not os.path.exists(args.weights):
        raise FileNotFoundError(f"Autonomous demo requires a valid weights file. Provide --weights <file>")

    agent = PPOAgent(
        state_shape=(args.grid_size, args.grid_size, 2),
        num_actions=env.num_actions
    )
    agent.model.load_weights(args.weights)
    state = env.reset()

    # Setup visualization
    if env.reward_type == 'pattern' and env.target_pattern is not None:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_target = fig.add_subplot(gs[:, 1])
        ax_probs = fig.add_subplot(gs[0, 2:])
        ax_value = fig.add_subplot(gs[1, 2])
        ax_metrics = fig.add_subplot(gs[1, 3])
        ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1)
        ax_target.set_title('Target Pattern')
        ax_target.set_xticks([])
        ax_target.set_yticks([])
    else:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_probs = fig.add_subplot(gs[0, 1:])
        ax_value = fig.add_subplot(gs[1, 1])
        ax_metrics = fig.add_subplot(gs[1, 2])

    grid_img = ax_main.imshow(env.ca_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                facecolor='none', edgecolor='cyan', linewidth=2)
    ax_main.add_patch(agent_patch)
    target_patch = plt.Rectangle((env.target_x - 0.5, env.target_y - 0.5), 2, 2,
                                 facecolor='none', edgecolor='red', linewidth=2, visible=env.has_target)
    ax_main.add_patch(target_patch)
    title_text = ax_main.set_title("Step: 0 | Agent")
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    action_labels = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
    action_probs, state_value = agent.get_action_probs_and_value(state)
    bars = ax_probs.bar(range(env.num_actions), action_probs, color='steelblue', alpha=0.8)
    ax_probs.set_ylim([0, 1])
    ax_probs.set_xticks(range(env.num_actions))
    ax_probs.set_xticklabels(action_labels, fontsize=8)
    ax_probs.set_ylabel('Probability')
    ax_probs.set_title('Action Distribution')
    ax_probs.grid(axis='y', alpha=0.25)

    value_text = ax_value.text(0.5, 0.5, f'V(s) = {state_value:.3f}',
                               ha='center', va='center', fontsize=18,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_value.set_xlim([0, 1])
    ax_value.set_ylim([0, 1])
    ax_value.set_title('State Value (Critic)')
    ax_value.axis('off')

    metrics_text = ax_metrics.text(0.05, 0.95, '', va='top', fontsize=10, family='monospace')
    ax_metrics.set_title('Metrics')
    ax_metrics.axis('off')

    value_history = deque(maxlen=200)
    entropy_history = deque(maxlen=200)

    def update(frame):
        nonlocal state
        action = agent.select_action(state)
        action_probs, state_value = agent.get_action_probs_and_value(state)
        clipped_probs = np.clip(action_probs, 1e-10, 1.0)
        entropy = -np.sum(clipped_probs * np.log(clipped_probs))
        value_history.append(state_value)
        entropy_history.append(entropy)

        next_state, reward, _, _ = env.step(action)
        state = next_state

        grid_img.set_data(env.ca_grid)
        agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))

        if env.has_target:
            target_patch.set_xy((env.target_x - 0.5, env.target_y - 0.5))
            target_patch.set_visible(True)
        else:
            target_patch.set_visible(False)

        for i, (bar, p) in enumerate(zip(bars, action_probs)):
            bar.set_height(p)
            bar.set_color('coral' if i == action else 'steelblue')

        value_text.set_text(f'V(s) = {state_value:.3f}')

        alive_count = int(np.sum(env.ca_grid))
        density = float(np.mean(env.ca_grid))
        avg_v = float(np.mean(value_history)) if value_history else 0.0
        avg_h = float(np.mean(entropy_history)) if entropy_history else 0.0

        metrics_str = (
            f"Step: {frame}\n"
            f"Action: {env.actions[action]}\n"
            f"Reward: {reward:.3f}\n"
            f"Alive: {alive_count}\n"
            f"Density: {density:.3f}\n"
            f"Entropy: {entropy:.3f}\n"
            f"Avg V: {avg_v:.3f}\n"
            f"Avg H: {avg_h:.3f}"
        )

        if env.reward_type == 'pattern' and env.target_pattern is not None:
            bce = env._calculate_pattern_bce()
            metrics_str += f"\nBCE: {bce:.4f}"
            title_text.set_text(f"Step: {frame} | Agent | BCE: {bce:.4f}")
        else:
            title_text.set_text(f"Step: {frame} | Agent")

        metrics_text.set_text(metrics_str)
        return [grid_img, agent_patch, target_patch] + list(bars)

    ani = animation.FuncAnimation(fig, update, frames=args.steps, interval=100, repeat=False, blit=False)
    print("\nAgent demo running (close window to stop).")
    plt.show(block=True)


def run_demo_manual(args):
    """Manual demo with optional recording."""
    import matplotlib as mpl
    mpl.rcParams['keymap.fullscreen'] = []

    print("--- Running Manual Demo ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward, max_steps=args.steps)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)

    agent = None
    if getattr(args, 'weights', None) and os.path.exists(args.weights):
        agent = PPOAgent(
            state_shape=(args.grid_size, args.grid_size, 2),
            num_actions=env.num_actions
        )
        try:
            agent.model.load_weights(args.weights)
            print(f"Loaded weights for manual-display from {args.weights}")
        except Exception as e:
            print(f"Failed loading weights for manual display: {e}")
            agent = None

    state = env.reset()

    if env.reward_type == 'pattern' and env.target_pattern is not None:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_target = fig.add_subplot(gs[:, 1])
        ax_probs = fig.add_subplot(gs[0, 2:])
        ax_metrics = fig.add_subplot(gs[1, 2:])
        ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1)
        ax_target.set_title('Target Pattern')
        ax_target.set_xticks([])
        ax_target.set_yticks([])
    else:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_probs = fig.add_subplot(gs[0, 1:])
        ax_metrics = fig.add_subplot(gs[1, 1:])
    
    grid_img = ax_main.imshow(env.ca_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                facecolor='none', edgecolor='cyan', linewidth=2)
    ax_main.add_patch(agent_patch)
    target_patch = plt.Rectangle((env.target_x - 0.5, env.target_y - 0.5), 2, 2,
                                 facecolor='none', edgecolor='red', linewidth=2, visible=env.has_target)
    ax_main.add_patch(target_patch)
    title_text = ax_main.set_title("Step: 0 | Manual")
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    action_labels = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
    if agent is not None:
        action_probs, state_value = agent.get_action_probs_and_value(state)
    else:
        action_probs = np.ones(env.num_actions) / env.num_actions
        state_value = 0.0

    bars = ax_probs.bar(range(env.num_actions), action_probs, color='steelblue', alpha=0.8)
    ax_probs.set_ylim([0, 1])
    ax_probs.set_xticks(range(env.num_actions))
    ax_probs.set_xticklabels(action_labels, fontsize=8)
    ax_probs.set_ylabel('Probability')
    ax_probs.set_title('Action Distribution')
    ax_probs.grid(axis='y', alpha=0.25)

    metrics_text = ax_metrics.text(0.05, 0.95, '', va='top', fontsize=10, family='monospace')
    ax_metrics.set_title('Metrics')
    ax_metrics.axis('off')

    recording = False
    states_logged, actions_logged, rewards_logged, next_states_logged, ts_logged = [], [], [], [], []
    step_counter = 0

    key_map = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3,
        ' ': 4,
    }
    hex_keys = list('0123456789abcdef')

    def on_key(event):
        nonlocal state, recording, step_counter
        if event.key is None:
            return
        key = event.key.lower()

        if key == 'r':
            recording = not recording
            print(f"Recording {'started' if recording else 'stopped'}.")
            return

        if key == 'p':
            if len(actions_logged) == 0:
                print("No manual data to save.")
            else:
                fname = f'manual_data_{int(time.time())}.npz'
                np.savez_compressed(fname,
                                    states=np.array(states_logged),
                                    actions=np.array(actions_logged),
                                    rewards=np.array(rewards_logged),
                                    next_states=np.array(next_states_logged),
                                    timestamps=np.array(ts_logged))
                print(f"Saved manual dataset to {fname} ({len(actions_logged)} samples)")
            return

        if key == 'q' or key == 'escape':
            plt.close()
            return

        if key in key_map:
            action = key_map[key]
        elif key in hex_keys:
            action = 5 + hex_keys.index(key)
        else:
            return

        if recording:
            states_logged.append(np.squeeze(state, axis=0).copy())

        next_state, reward, _, _ = env.step(action)

        if recording:
            actions_logged.append(action)
            rewards_logged.append(reward)
            next_states_logged.append(np.squeeze(next_state, axis=0).copy())
            ts_logged.append(time.time())

        state = next_state
        step_counter += 1

        grid_img.set_data(env.ca_grid)
        agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))

        if env.has_target:
            target_patch.set_xy((env.target_x - 0.5, env.target_y - 0.5))
            target_patch.set_visible(True)
        else:
            target_patch.set_visible(False)

        if agent is not None:
            action_probs, state_value = agent.get_action_probs_and_value(state)
        else:
            action_probs = np.ones(env.num_actions) / env.num_actions
            state_value = 0.0

        for i, (bar, p) in enumerate(zip(bars, action_probs)):
            bar.set_height(p)
            bar.set_color('coral' if i == action else 'steelblue')

        alive_count = int(np.sum(env.ca_grid))
        density = float(np.mean(env.ca_grid))
        metrics_str = (
            f"Step: {step_counter}\n"
            f"Action: {env.actions[action]}\n"
            f"Reward: {reward:.3f}\n"
            f"Alive: {alive_count}\n"
            f"Density: {density:.3f}\n"
            f"Recording: {recording}\n"
            f"V(s): {state_value:.3f}"
        )
        if env.reward_type == 'pattern':
            bce = env._calculate_pattern_bce()
            metrics_str += f"\nBCE: {bce:.4f}"
            title_text.set_text(f"Step: {step_counter} | Manual | BCE: {bce:.4f}")
        else:
            title_text.set_text(f"Step: {step_counter} | Manual")

        metrics_text.set_text(metrics_str)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\nManual Control Enabled:")
    print("- Arrow Keys to move, Space to do nothing")
    print("- Hex keys 0-f to write 2x2 patterns")
    print("- r to start/stop recording, p to save, q to quit\n")

    plt.show(block=True)

    if len(actions_logged) > 0:
        fname = f'manual_data_{int(time.time())}.npz'
        np.savez_compressed(fname,
                            states=np.array(states_logged),
                            actions=np.array(actions_logged),
                            rewards=np.array(rewards_logged),
                            next_states=np.array(next_states_logged),
                            timestamps=np.array(ts_logged))
        print(f"Manual demo finished. Saved {len(actions_logged)} samples to {fname}")


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced RL in Cellular Automata Environment with PPO.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select mode')

    # --- Training Arguments ---
    train_parser = subparsers.add_parser('train', help='Run PPO training.')
    train_parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes.')
    train_parser.add_argument('--rollout-steps', type=int, default=10, help='Steps per rollout/episode.')
    train_parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate.')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    train_parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda.')
    train_parser.add_argument('--clip-eps', type=float, default=0.2, help='PPO clip epsilon.')
    train_parser.add_argument('--entropy-coef', type=float, default=0.05, help='Entropy coefficient.')
    train_parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient.')
    train_parser.add_argument('--ppo-epochs', type=int, default=4, help='PPO update epochs per rollout.')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Minibatch size for PPO updates.')
    train_parser.add_argument('--grid-size', type=int, default=12, help='Size of the CA grid.')
    train_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'])
    train_parser.add_argument('--reward', type=str, default='entropy', 
                             choices=['entropy', 'maxwell_demon', 'target_practice', 'pattern'])
    train_parser.add_argument('--pattern-file', type=str, default=None, help='Pattern file for training.')
    train_parser.add_argument('--live-plot', type=int, nargs='?', const=5, default=None, 
                             help='Live plotting frequency (0=every step, N=every N episodes).')
    train_parser.add_argument('--pretrained-weights', type=str, default=None, help='Pretrained weights path.')

    # --- Supervised Learning Arguments ---
    sup_parser = subparsers.add_parser('supervised', help='Supervised pretraining.')
    sup_parser.add_argument('--data-file', type=str, required=True, help='Path to .npz data file.')
    sup_parser.add_argument('--epochs', type=int, default=10, help='Training epochs.')
    sup_parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    sup_parser.add_argument('--grid-size', type=int, default=12)
    sup_parser.add_argument('--rules', type=str, default='conway', choices=['conway','seeds','maze'])
    sup_parser.add_argument('--reward', type=str, default='entropy')
    sup_parser.add_argument('--init-weights', type=str, default=None)
    sup_parser.add_argument('--save-weights', type=str, default='supervised_weights_final.weights.h5')
    sup_parser.add_argument('--save-every', type=int, default=5)
    sup_parser.add_argument('--lr', type=float, default=1e-4)
    sup_parser.add_argument('--entropy-coef', type=float, default=0.0)
    sup_parser.add_argument('--gamma', type=float, default=0.99)
    sup_parser.add_argument('--value-coef', type=float, default=0.5)

    # --- Demo Arguments ---
    demo_parser = subparsers.add_parser('demo', help='Run autonomous demo.')
    demo_parser.add_argument('--weights', type=str, default='ca_agent_weights_final.weights.h5')
    demo_parser.add_argument('--steps', type=int, default=10)
    demo_parser.add_argument('--grid-size', type=int, default=12)
    demo_parser.add_argument('--rules', type=str, default='conway', choices=['conway','seeds','maze'])
    demo_parser.add_argument('--reward', type=str, default='entropy')
    demo_parser.add_argument('--pattern-file', type=str, default=None)

    # --- Manual Demo ---
    manual_parser = subparsers.add_parser('manual', help='Manual play with recording.')
    manual_parser.add_argument('--grid-size', type=int, default=12)
    manual_parser.add_argument('--rules', type=str, default='conway', choices=['conway','seeds','maze'])
    manual_parser.add_argument('--reward', type=str, default='entropy')
    manual_parser.add_argument('--pattern-file', type=str, default=None)
    manual_parser.add_argument('--steps', type=int, default=10)
    manual_parser.add_argument('--weights', type=str, default=None, help='Optional weights for policy display')

    # --- Pattern Creator Arguments ---
    pattern_parser = subparsers.add_parser('create_pattern', help='Interactive pattern creator.')
    pattern_parser.add_argument('--grid-size', type=int, default=12)

    args = parser.parse_args()

    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'supervised':
        supervised_pretrain(args)
    elif args.mode == 'demo':
        run_demo_auto(args)
    elif args.mode == 'manual':
        run_demo_manual(args)
    elif args.mode == 'create_pattern':
        interactive_pattern_creator(args)