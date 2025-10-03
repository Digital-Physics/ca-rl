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
                 reward_type='entropy', target_pattern=None):
        self.grid_size = grid_size
        self.initial_density = initial_density
        self.rules_name = rules_name
        self.reward_type = reward_type

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
                # Default simple pattern if none provided
                self.target_pattern = None
        else:
            self.target_pattern = None

        self.reset()

    def _apply_ca_rules_fast(self, grid):
        """Fast CA rule application using convolution."""
        # Use 'wrap' mode for toroidal boundary
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
            binary_cross_entropy = self._calculate_pattern_bce()
            reward = -binary_cross_entropy
            if binary_cross_entropy < 0.01:
                reward += 10

        done = False
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
        
        bce = -np.mean(target * np.log(current) + (1 - target) * np.log(1 - current))
        
        return bce

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


# --- Actor-Critic Agent ---
class ActorCriticAgent:
    """
    The Actor-Critic agent with enhanced metrics extraction.
    """
    def __init__(self, state_shape, num_actions, learning_rate=0.0001, gamma=0.95, entropy_coef=0.01):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self._build_networks()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.huber_loss = keras.losses.Huber()

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

    def get_action_probs_and_value(self, state):
        """Get action probabilities and state value for visualization."""
        logits, value = self.model(state)
        action_probs = tf.nn.softmax(logits).numpy()[0]
        value = value.numpy()[0, 0]
        return action_probs, value

    @tf.function
    def train_step(self, state, action, reward, next_state, done):
        """Performs a single training step with entropy regularization."""
        with tf.GradientTape() as tape:
            action_logits, state_value = self.model(state)
            _, next_state_value = self.model(next_state)

            # Critic loss
            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            td_target = reward + self.gamma * next_state_value * (1 - tf.cast(done, tf.float32))
            advantage = td_target - state_value
            critic_loss = self.huber_loss(tf.expand_dims(td_target, 0), tf.expand_dims(state_value, 0))

            # Actor loss with entropy regularization
            action_indices = tf.stack([tf.range(state.shape[0], dtype=tf.int32), tf.cast(action, tf.int32)], axis=1)
            log_probs = tf.nn.log_softmax(action_logits)
            action_log_probs = tf.gather_nd(log_probs, action_indices)
            
            # Entropy bonus for exploration
            action_probs = tf.nn.softmax(action_logits)
            entropy = -tf.reduce_sum(action_probs * log_probs, axis=-1)
            
            actor_loss = -action_log_probs * tf.stop_gradient(advantage) - self.entropy_coef * entropy

            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Return metrics for monitoring
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'advantage': advantage,
            'entropy': entropy,
            'td_error': tf.abs(advantage)
        }

# Are there too many graphs being created
def train_agent(args):
    """Main training loop with enhanced live visualization."""
    print("--- Starting Training ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward)
    agent = ActorCriticAgent(
        state_shape=(args.grid_size, args.grid_size, 2),
        num_actions=env.num_actions,
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef
    )

    # Metrics tracking
    total_rewards = []
    actor_losses = []
    critic_losses = []
    advantages = []
    entropies = []
    td_errors = []
    alive_cells = []
    density_values = []
    policy_entropies = []

    # Setup live plotting if enabled
    if args.live_plot:
        plt.ion()
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
        
        # Top row: CA grid, action probabilities, value function, metrics display
        ax_grid = fig.add_subplot(gs[0, 0])
        ax_probs = fig.add_subplot(gs[0, 1:3])
        ax_value = fig.add_subplot(gs[0, 3])
        
        # Middle row: Training metrics
        ax_reward = fig.add_subplot(gs[1, 0])
        ax_loss = fig.add_subplot(gs[1, 1])
        ax_entropy_metric = fig.add_subplot(gs[1, 2])
        ax_td = fig.add_subplot(gs[1, 3])
        
        # Bottom row: More metrics
        ax_advantage = fig.add_subplot(gs[2, 0])
        ax_alive = fig.add_subplot(gs[2, 1])
        ax_density = fig.add_subplot(gs[2, 2])
        ax_policy_entropy = fig.add_subplot(gs[2, 3])
        
        fig.suptitle('Training Progress (Live)', fontsize=16, fontweight='bold')
        
        # Setup CA grid visualization
        grid_img = ax_grid.imshow(env.ca_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                    facecolor='none', edgecolor='cyan', linewidth=2)
        ax_grid.add_patch(agent_patch)
        ax_grid.set_title('Current State', fontweight='bold')
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        
        # Setup action probability bars
        action_labels = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
        bars = ax_probs.bar(range(env.num_actions), np.zeros(env.num_actions), 
                            color='steelblue', alpha=0.7)
        ax_probs.set_ylim([0, 1])
        ax_probs.set_xticks(range(env.num_actions))
        ax_probs.set_xticklabels(action_labels, fontsize=8)
        ax_probs.set_ylabel('Probability')
        ax_probs.set_title('Action Distribution (Current State)', fontweight='bold')
        ax_probs.grid(axis='y', alpha=0.3)
        
        # Setup value function display
        value_text = ax_value.text(0.5, 0.5, 'V(s) = 0.000',
                                   ha='center', va='center', fontsize=18,
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax_value.set_xlim([0, 1])
        ax_value.set_ylim([0, 1])
        ax_value.set_title('State Value (Critic)', fontweight='bold')
        ax_value.axis('off')
        
        # Setup metric plots
        lines = {
            'reward': ax_reward.plot([], [], 'b-', linewidth=2)[0],
            'actor_loss': ax_loss.plot([], [], 'r-', label='Actor', linewidth=2)[0],
            'critic_loss': ax_loss.plot([], [], 'g-', label='Critic', linewidth=2)[0],
            'entropy': ax_entropy_metric.plot([], [], 'orange', linewidth=2)[0],
            'td_error': ax_td.plot([], [], 'brown', linewidth=2)[0],
            'advantage': ax_advantage.plot([], [], 'purple', linewidth=2)[0],
            'alive': ax_alive.plot([], [], 'teal', linewidth=2)[0],
            'density': ax_density.plot([], [], 'darkgreen', linewidth=2)[0],
            'policy_entropy': ax_policy_entropy.plot([], [], 'crimson', linewidth=2)[0],
        }
        
        ax_reward.set_title('Episode Reward', fontweight='bold')
        ax_reward.set_xlabel('Episode')
        ax_reward.set_ylabel('Total Reward')
        ax_reward.grid(True, alpha=0.3)
        
        ax_loss.set_title('Losses', fontweight='bold')
        ax_loss.set_xlabel('Episode')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        ax_entropy_metric.set_title('Training Entropy', fontweight='bold')
        ax_entropy_metric.set_xlabel('Episode')
        ax_entropy_metric.set_ylabel('Entropy')
        ax_entropy_metric.grid(True, alpha=0.3)
        
        ax_td.set_title('TD Error', fontweight='bold')
        ax_td.set_xlabel('Episode')
        ax_td.set_ylabel('|TD Error|')
        ax_td.grid(True, alpha=0.3)
        
        ax_advantage.set_title('Average Advantage', fontweight='bold')
        ax_advantage.set_xlabel('Episode')
        ax_advantage.set_ylabel('Advantage')
        ax_advantage.grid(True, alpha=0.3)
        
        ax_alive.set_title('Avg Alive Cells', fontweight='bold')
        ax_alive.set_xlabel('Episode')
        ax_alive.set_ylabel('Count')
        ax_alive.grid(True, alpha=0.3)
        
        ax_density.set_title('Grid Density', fontweight='bold')
        ax_density.set_xlabel('Episode')
        ax_density.set_ylabel('Density')
        ax_density.grid(True, alpha=0.3)
        
        ax_policy_entropy.set_title('Policy Entropy (H)', fontweight='bold')
        ax_policy_entropy.set_xlabel('Episode')
        ax_policy_entropy.set_ylabel('H(π)')
        ax_policy_entropy.grid(True, alpha=0.3)

    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0.0
        ep_metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'advantage': 0.0,
            'entropy': 0.0,
            'td_error': 0.0,
            'alive': 0.0,
            'density': 0.0,
            'policy_entropy': 0.0
        }

        pbar = tqdm(range(args.steps), desc=f"Episode {episode + 1}/{args.episodes}")
        for step in pbar:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            metrics = agent.train_step(
                state,
                np.array([action]),
                np.array([reward], dtype=np.float32),
                next_state,
                np.array([done])
            )

            # Calculate policy entropy for current state
            action_probs_current, _ = agent.get_action_probs_and_value(state)
            policy_entropy = -np.sum(action_probs_current * np.log(action_probs_current + 1e-10))

            state = next_state
            episode_reward += float(reward)
            
            # Accumulate metrics
            ep_metrics['actor_loss'] += float(metrics['actor_loss'].numpy())
            ep_metrics['critic_loss'] += float(metrics['critic_loss'].numpy())
            ep_metrics['advantage'] += float(metrics['advantage'].numpy())
            ep_metrics['entropy'] += float(metrics['entropy'].numpy())
            ep_metrics['td_error'] += float(metrics['td_error'].numpy())
            ep_metrics['alive'] += np.sum(env.ca_grid)
            ep_metrics['density'] += np.mean(env.ca_grid)
            ep_metrics['policy_entropy'] += policy_entropy

            pbar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Entropy': f'{ep_metrics["entropy"] / (step + 1):.3f}',
                'Value': f'{agent.get_action_probs_and_value(state)[1]:.2f}'
            })

        # Store episode metrics
        total_rewards.append(episode_reward)
        actor_losses.append(ep_metrics['actor_loss'] / args.steps)
        critic_losses.append(ep_metrics['critic_loss'] / args.steps)
        advantages.append(ep_metrics['advantage'] / args.steps)
        entropies.append(ep_metrics['entropy'] / args.steps)
        td_errors.append(ep_metrics['td_error'] / args.steps)
        # alive_cells.append(ep_metrics['alive'] / args.steps)
        # density_values.append(ep_metrics['density'] / args.steps)
        # policy_entropies.append(ep_metrics['policy_entropy'] / args.steps)
        alive_cells.append(float(ep_metrics['alive'] / args.steps))
        density_values.append(float(ep_metrics['density'] / args.steps))
        policy_entropies.append(float(ep_metrics['policy_entropy'] / args.steps))

        # Update live plot
        if args.live_plot and (episode + 1) % 5 == 0:
            episodes_range = range(1, episode + 2)
            
            # Update grid and agent position
            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            
            # Get current action probs and value
            action_probs, current_value = agent.get_action_probs_and_value(state)
            
            # Update action probability bars
            for bar, prob in zip(bars, action_probs):
                bar.set_height(prob)
            
            # Update value display
            value_text.set_text(f'V(s) = {current_value:.3f}')
            
            # Update all metric lines
            lines['reward'].set_data(episodes_range, total_rewards)
            lines['actor_loss'].set_data(episodes_range, actor_losses)
            lines['critic_loss'].set_data(episodes_range, critic_losses)
            lines['entropy'].set_data(episodes_range, entropies)
            lines['td_error'].set_data(episodes_range, td_errors)
            lines['advantage'].set_data(episodes_range, advantages)
            lines['alive'].set_data(episodes_range, alive_cells)
            lines['density'].set_data(episodes_range, density_values)
            lines['policy_entropy'].set_data(episodes_range, policy_entropies)
            
            # Rescale all axes
            for ax in [ax_reward, ax_loss, ax_entropy_metric, ax_td, 
                       ax_advantage, ax_alive, ax_density, ax_policy_entropy]:
                ax.relim()
                ax.autoscale_view()
            
            plt.pause(0.01)

        # Save model weights periodically
        if (episode + 1) % 25 == 0:
            agent.model.save_weights(f'ca_agent_weights_{episode+1}.weights.h5')
            print(f"\nSaved weights at episode {episode+1}")

    agent.model.save_weights('ca_agent_weights_final.weights.h5')
    print("\n--- Training Complete ---")
    
    # Save training metrics
    metrics_data = {
        'rewards': total_rewards,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'advantages': advantages,
        'entropies': entropies,
        'td_errors': td_errors,
        'alive_cells': alive_cells,
        'density_values': density_values,
        'policy_entropies': policy_entropies
    }
    print(metrics_data)
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_data, f)
    print("Training metrics saved to training_metrics.json")

    # Final plot
    if args.live_plot:
        plt.ioff()
    
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 3, 1)
    plt.plot(total_rewards, linewidth=2)
    plt.title('Total Reward per Episode', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 2)
    plt.plot(actor_losses, label='Actor Loss', linewidth=2)
    plt.plot(critic_losses, label='Critic Loss', linewidth=2)
    plt.title('Average Loss per Episode', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 3)
    plt.plot(advantages, linewidth=2, color='purple')
    plt.title('Average Advantage', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Advantage')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 4)
    plt.plot(entropies, linewidth=2, color='orange')
    plt.title('Training Entropy', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 5)
    plt.plot(td_errors, linewidth=2, color='brown')
    plt.title('TD Error', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('|TD Error|')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 6)
    plt.plot(alive_cells, linewidth=2, color='teal')
    plt.title('Average Alive Cells', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 7)
    plt.plot(density_values, linewidth=2, color='darkgreen')
    plt.title('Grid Density', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 8)
    plt.plot(policy_entropies, linewidth=2, color='crimson')
    plt.title('Policy Entropy (H)', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('H(π)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 9)
    # Plot smoothed reward with rolling average
    if len(total_rewards) > 10:
        window = min(10, len(total_rewards))
        smoothed = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(total_rewards)), smoothed, linewidth=2, color='blue')
        plt.title('Smoothed Reward (10-ep avg)', fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Avg Reward')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("Final plot saved to training_results.png")
    plt.show()


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
    
    plt.show()


def run_demo(args):
    """Runs a visualization with enhanced metrics display."""
    import matplotlib as mpl
    mpl.rcParams['keymap.fullscreen'] = []

    print("--- Starting Demo ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward)
    
    # Load custom pattern if specified
    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)
    
    agent = ActorCriticAgent(
        state_shape=(args.grid_size, args.grid_size, 2),
        num_actions=env.num_actions
    )
    
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        agent.model.load_weights(args.weights)
        manual_mode = False
    else:
        print(f"Weights file not found: {args.weights}. Starting in manual control mode.")
        manual_mode = True

    state = env.reset()
    
    # Setup figure with enhanced visualizations
    if env.reward_type == 'pattern':
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
    
    # Main grid display
    grid_img = ax_main.imshow(env.ca_grid, cmap='binary', vmin=0, vmax=1)
    agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2, 
                                facecolor='none', edgecolor='cyan', linewidth=2)
    ax_main.add_patch(agent_patch)
    target_patch = plt.Rectangle((env.target_x - 0.5, env.target_y - 0.5), 2, 2,
                                 facecolor='none', edgecolor='red', linewidth=2, visible=False)
    ax_main.add_patch(target_patch)
    title_text = ax_main.set_title("Step: 0 | Mode: " + ("Manual" if manual_mode else "Agent"))
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    
    # Action probability bar chart
    action_labels = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
    if not manual_mode:
        action_probs, state_value = agent.get_action_probs_and_value(state)
    else:
        action_probs = np.ones(env.num_actions) / env.num_actions
        state_value = 0.0
    
    bars = ax_probs.bar(range(env.num_actions), action_probs, color='steelblue', alpha=0.7)
    ax_probs.set_ylim([0, 1])
    ax_probs.set_xticks(range(env.num_actions))
    ax_probs.set_xticklabels(action_labels, fontsize=8)
    ax_probs.set_ylabel('Probability')
    ax_probs.set_title('Action Distribution')
    ax_probs.grid(axis='y', alpha=0.3)
    
    # Value function display
    value_text = ax_value.text(0.5, 0.5, f'V(s) = {state_value:.3f}', 
                               ha='center', va='center', fontsize=20, 
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_value.set_xlim([0, 1])
    ax_value.set_ylim([0, 1])
    ax_value.set_title('State Value (Critic)')
    ax_value.axis('off')
    
    # Additional metrics
    metrics_text = ax_metrics.text(0.1, 0.9, '', va='top', fontsize=10, family='monospace')
    ax_metrics.set_xlim([0, 1])
    ax_metrics.set_ylim([0, 1])
    ax_metrics.set_title('Metrics')
    ax_metrics.axis('off')
    
    # Tracking for metrics
    value_history = deque(maxlen=100)
    entropy_history = deque(maxlen=100)
    
    current_action = 4  # do nothing
    step_requested = {'flag': False}

    key_map = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3,
        ' ': 4,
    }
    hex_keys = list('0123456789abcdef')

    def on_press(event):
        if event.key is None:
            return
        key = event.key.lower()
        nonlocal current_action

        if key in key_map:
            current_action = key_map[key]
            step_requested['flag'] = True
        elif key in hex_keys:
            pattern_index = hex_keys.index(key)
            current_action = 5 + pattern_index
            step_requested['flag'] = True

    fig.canvas.mpl_connect('key_press_event', on_press)

    if manual_mode:
        print("\nManual Control Enabled:")
        print("- Arrow Keys to move")
        print("- Spacebar to do nothing")
        print("- Hex keys 0-f to write patterns\n")

        step = 0
        while step < args.steps:
            plt.pause(0.01)
            if not step_requested['flag']:
                continue
            action = current_action
            current_action = 4
            step_requested['flag'] = False

            next_state, reward, _, _ = env.step(action)
            state = next_state

            # Update visualizations
            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))

            if env.has_target:
                target_patch.set_xy((env.target_x - 0.5, env.target_y - 0.5))
                target_patch.set_visible(True)
            else:
                target_patch.set_visible(False)

            # Update metrics
            alive_count = np.sum(env.ca_grid)
            density = np.mean(env.ca_grid)
            
            metrics_str = f"Step: {step}\n"
            metrics_str += f"Action: {env.actions[action]}\n"
            metrics_str += f"Reward: {reward:.3f}\n"
            metrics_str += f"Alive: {alive_count}\n"
            metrics_str += f"Density: {density:.3f}\n"
            
            if env.reward_type == 'pattern':
                bce = env._calculate_pattern_bce()
                metrics_str += f"BCE: {bce:.4f}"
                title_text.set_text(f"Step: {step} | Manual | BCE: {bce:.4f}")
            else:
                title_text.set_text(f"Step: {step} | Manual")
            
            metrics_text.set_text(metrics_str)
            fig.canvas.draw_idle()
            step += 1

        print("Manual demo finished.")
        plt.show(block=True)
    else:
        def update(frame):
            nonlocal state
            action = agent.select_action(state)
            next_state, reward, _, _ = env.step(action)
            
            # Get current metrics
            action_probs, state_value = agent.get_action_probs_and_value(state)
            entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
            
            value_history.append(state_value)
            entropy_history.append(entropy)
            
            state = next_state

            # Update grid and agent
            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            
            if env.has_target:
                target_patch.set_xy((env.target_x - 0.5, env.target_y - 0.5))
                target_patch.set_visible(True)
            else:
                target_patch.set_visible(False)

            # Update action probabilities
            for bar, prob in zip(bars, action_probs):
                bar.set_height(prob)
                # Highlight selected action
                if bars.patches.index(bar) == action:
                    bar.set_color('coral')
                else:
                    bar.set_color('steelblue')
            
            # Update value display
            value_text.set_text(f'V(s) = {state_value:.3f}')
            
            # Update metrics
            alive_count = np.sum(env.ca_grid)
            density = np.mean(env.ca_grid)
            avg_value = np.mean(value_history) if value_history else 0
            avg_entropy = np.mean(entropy_history) if entropy_history else 0
            
            metrics_str = f"Step: {frame}\n"
            metrics_str += f"Action: {env.actions[action]}\n"
            metrics_str += f"Reward: {reward:.3f}\n"
            metrics_str += f"Alive: {alive_count}\n"
            metrics_str += f"Density: {density:.3f}\n"
            metrics_str += f"Entropy: {entropy:.3f}\n"
            metrics_str += f"Avg V(s): {avg_value:.3f}\n"
            metrics_str += f"Avg H: {avg_entropy:.3f}"
            
            if env.reward_type == 'pattern':
                bce = env._calculate_pattern_bce()
                metrics_str += f"\nBCE: {bce:.4f}"
                title_text.set_text(f"Step: {frame} | Agent | BCE: {bce:.4f}")
            else:
                title_text.set_text(f"Step: {frame} | Agent")
            
            metrics_text.set_text(metrics_str)
            
            return grid_img, agent_patch, target_patch, title_text

        ani = animation.FuncAnimation(fig, update, frames=args.steps, 
                                     interval=100, repeat=False, blit=False)
        print("\nAgent demo running (close window to stop).")
        plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced RL in Cellular Automata Environment.")
    subparsers = parser.add_subparsers(dest='mode', required=True, 
                                      help='Select mode: train, demo, or create_pattern')

    # --- Training Arguments ---
    train_parser = subparsers.add_parser('train', help='Run the training loop.')
    train_parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes.')
    train_parser.add_argument('--steps', type=int, default=200, help='Number of steps per episode.')
    train_parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    train_parser.add_argument('--entropy-coef', type=float, default=0.01, 
                             help='Entropy regularization coefficient.')
    train_parser.add_argument('--grid-size', type=int, default=12, help='Size of the CA grid.')
    train_parser.add_argument('--rules', type=str, default='conway', 
                             choices=['conway', 'seeds', 'maze'], help='CA rule set.')
    train_parser.add_argument('--reward', type=str, default='entropy', 
                             choices=['entropy', 'maxwell_demon', 'target_practice', 'pattern'], 
                             help='Reward function.')
    train_parser.add_argument('--pattern-file', type=str, default=None, 
                            help='Load custom pattern from file for training.')
    train_parser.add_argument('--live-plot', action='store_true', 
                             help='Enable live plotting during training.')

    # --- Demo Arguments ---
    demo_parser = subparsers.add_parser('demo', help='Run a visual demonstration.')
    demo_parser.add_argument('--weights', type=str, default='ca_agent_weights_final.weights.h5', 
                            help='Path to saved model weights.')
    demo_parser.add_argument('--steps', type=int, default=500, help='Number of demo steps.')
    demo_parser.add_argument('--grid-size', type=int, default=12, help='Size of the CA grid.')
    demo_parser.add_argument('--rules', type=str, default='conway', 
                            choices=['conway', 'seeds', 'maze'], help='CA rule set.')
    demo_parser.add_argument('--reward', type=str, default='entropy', 
                            choices=['entropy', 'maxwell_demon', 'target_practice', 'pattern'], 
                            help='Reward function.')
    demo_parser.add_argument('--pattern-file', type=str, default=None, 
                            help='Load custom pattern from file.')

    # --- Pattern Creator Arguments ---
    pattern_parser = subparsers.add_parser('create_pattern', 
                                          help='Interactive pattern creation tool.')
    pattern_parser.add_argument('--grid-size', type=int, default=12, 
                               help='Size of the pattern grid.')

    args = parser.parse_args()

    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'create_pattern':
        interactive_pattern_creator(args)