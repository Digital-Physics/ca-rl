# import matplotlib
# matplotlib.use('TkAgg') 
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


# Enable interactive mode once for drawing some images during training
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
    def __init__(self, state_shape, num_actions, learning_rate=0.001, gamma=0.95, entropy_coef=0.01):
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
        """
        Performs a single training step (Actor-Critic) with Advantage Clipping
        AND Entropy Regularization.
        """
        # Define clipping range (to limit learning step)
        ADVANTAGE_CLIP = 1.5
        
        with tf.GradientTape() as tape:
            action_logits, state_value = self.model(state)
            _, next_state_value = self.model(next_state)

            # Critic loss (remove dim)
            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            td_target = reward + self.gamma * next_state_value * (1 - tf.cast(done, tf.float32))
            advantage = td_target - state_value
            
            critic_loss = self.huber_loss(tf.expand_dims(td_target, 0), tf.expand_dims(state_value, 0))

            # Actor loss
            action_indices = tf.stack([tf.range(state.shape[0], dtype=tf.int32), tf.cast(action, tf.int32)], axis=1)
            log_probs = tf.nn.log_softmax(action_logits)
            action_log_probs = tf.gather_nd(log_probs, action_indices)
            
            # Clip the advantage to stabilize training
            clipped_advantage = tf.clip_by_value(advantage, 
                                                clip_value_min=-ADVANTAGE_CLIP, 
                                                clip_value_max=ADVANTAGE_CLIP)
            
            # Calculate entropy to encourage exploration
            action_probs = tf.nn.softmax(action_logits)
            # sane as policy entropy
            entropy = -tf.reduce_sum(action_probs * log_probs, axis=-1)
            
            # Actor loss now includes both clipped advantage and the entropy bonus
            actor_loss = -action_log_probs * tf.stop_gradient(clipped_advantage) - self.entropy_coef * entropy

            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Return metrics for monitoring
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'advantage': clipped_advantage, # Return the clipped value for monitoring
            'entropy': entropy,
            'td_error': tf.abs(advantage) # TD error uses the unclipped advantage
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

    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward)
    agent = ActorCriticAgent(
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
                    # Bootstrap target: r + γ * V(next)
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
    """Main training loop with enhanced live visualization and metrics."""
    print("--- Starting Training ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward)
    
    # Load pattern if reward_type is 'pattern'
    if args.reward == 'pattern' and args.pattern_file:
        env.load_pattern(args.pattern_file)

    agent = ActorCriticAgent(
        state_shape=(args.grid_size, args.grid_size, 2),
        num_actions=env.num_actions,
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef
    )

    # If provided, load supervised / pretrained weights to initialize RL
    if getattr(args, 'pretrained_weights', None):
        if os.path.exists(args.pretrained_weights):
            print(f"Loading pretrained weights from {args.pretrained_weights}")
            agent.model.load_weights(args.pretrained_weights)
        else:
            print(f"Pretrained weights not found at {args.pretrained_weights} (continuing without).")

    # Metrics tracking
    total_rewards = []
    actor_losses = []
    critic_losses = []
    advantages = []
    entropies = []
    td_errors = []
    density_values = []
    policy_entropies = []
    avg_state_values = [] 

    # Setup live plotting if enabled
    if args.live_plot:
        plt.ion()
        # Adjusted layout: 3 rows x 4 columns
        fig = plt.figure(figsize=(20, 15))
        # New GridSpec: 4 rows x 4 columns, but now Row 4 only has 2 plots, then empty space
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35)
        
        # Row 1: CA grid, Target Pattern, Action Probabilities, Value Display
        ax_grid = fig.add_subplot(gs[0, 0])
        
        # ADDED TARGET PATTERN LOGIC
        if env.reward_type == 'pattern' and env.target_pattern is not None:
            ax_target = fig.add_subplot(gs[0, 1])
            ax_probs = fig.add_subplot(gs[0, 2])
            ax_value_display = fig.add_subplot(gs[0, 3])
        else:
            ax_target = None
            ax_probs = fig.add_subplot(gs[0, 1:3])
            ax_value_display = fig.add_subplot(gs[0, 3])
            
        # Row 2: Training metrics (Reward, Losses, Entropy, TD Error)
        ax_reward = fig.add_subplot(gs[1, 0])
        ax_loss = fig.add_subplot(gs[1, 1])
        ax_entropy_metric = fig.add_subplot(gs[1, 2])
        ax_td = fig.add_subplot(gs[1, 3])
        
        # Row 3: More metrics (Advantage, Density, Policy Entropy, Avg V(s))
        ax_advantage = fig.add_subplot(gs[2, 0])
        ax_density = fig.add_subplot(gs[2, 1]) 
        ax_policy_entropy = fig.add_subplot(gs[2, 2]) 
        ax_avg_value = fig.add_subplot(gs[2, 3]) # ADDED
        
        # Row 4: Empty (3x3 grid makes a cleaner layout now)
        # We will use a 3x4 grid spec for better spacing now that grad norm is gone.
        # Rerunning GridSpec for cleaner 3x4 layout:
        plt.close(fig) # Close the old figure
        fig = plt.figure(figsize=(18, 14)) # Slightly smaller, more compact
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
        
        # Row 1: CA grid, Target Pattern, Action Probabilities, Value Display
        ax_grid = fig.add_subplot(gs[0, 0])
        if env.reward_type == 'pattern' and env.target_pattern is not None:
            ax_target = fig.add_subplot(gs[0, 1])
            ax_probs = fig.add_subplot(gs[0, 2])
            ax_value_display = fig.add_subplot(gs[0, 3])
        else:
            ax_target = None
            ax_probs = fig.add_subplot(gs[0, 1:3])
            ax_value_display = fig.add_subplot(gs[0, 3])
            
        # Row 2: Training metrics (Reward, Losses, Entropy, TD Error)
        ax_reward = fig.add_subplot(gs[1, 0])
        ax_loss = fig.add_subplot(gs[1, 1])
        ax_entropy_metric = fig.add_subplot(gs[1, 2])
        ax_td = fig.add_subplot(gs[1, 3])
        
        # Row 3: More metrics (Advantage, Density, Policy Entropy, Avg V(s))
        ax_advantage = fig.add_subplot(gs[2, 0])
        ax_density = fig.add_subplot(gs[2, 1]) 
        ax_policy_entropy = fig.add_subplot(gs[2, 2]) 
        ax_avg_value = fig.add_subplot(gs[2, 3]) # ADDED
        
        fig.suptitle('Training Progress (Live)', fontsize=16, fontweight='bold')
        
        # Setup CA grid visualization
        grid_img = ax_grid.imshow(env.ca_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                    facecolor='none', edgecolor='cyan', linewidth=2)
        ax_grid.add_patch(agent_patch)
        ax_grid.set_title(f"Last State In Episode", fontweight='bold')
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])

        # Setup Target Pattern Visualization (NEW)
        if ax_target is not None:
            ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
            ax_target.set_title('Target Pattern', fontweight='bold')
            ax_target.set_xticks([])
            ax_target.set_yticks([])

        # Setup action probability bars
        action_labels = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
        bars = ax_probs.bar(range(env.num_actions), np.zeros(env.num_actions), 
                            color='steelblue', alpha=0.7)
        ax_probs.set_ylim([0, 1])
        ax_probs.set_xticks(range(env.num_actions))
        ax_probs.set_xticklabels(action_labels, fontsize=8)
        ax_probs.set_ylabel('Probability')
        ax_probs.set_title('Action Distribution (Last State)', fontweight='bold')
        ax_probs.grid(axis='y', alpha=0.3)
        
        # Setup value function display (UPDATED)
        value_text = ax_value_display.text(0.5, 0.5, 'V(s) = 0.000',
                                   ha='center', va='center', fontsize=18,
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax_value_display.set_xlim([0, 1])
        ax_value_display.set_ylim([0, 1])
        ax_value_display.set_title('State Value (Critic) - Last Step', fontweight='bold')
        ax_value_display.axis('off')
        
        # Setup metric plots
        lines = {
            'reward': ax_reward.plot([], [], 'b-', linewidth=2)[0],
            'actor_loss': ax_loss.plot([], [], 'r-', label='Actor', linewidth=2)[0],
            'critic_loss': ax_loss.plot([], [], 'g-', label='Critic', linewidth=2)[0],
            'entropy': ax_entropy_metric.plot([], [], 'orange', linewidth=2)[0],
            'td_error': ax_td.plot([], [], 'brown', linewidth=2)[0],
            'advantage': ax_advantage.plot([], [], 'purple', linewidth=2)[0],
            'density': ax_density.plot([], [], 'darkgreen', linewidth=2)[0],
            'policy_entropy': ax_policy_entropy.plot([], [], 'crimson', linewidth=2)[0],
            'avg_value': ax_avg_value.plot([], [], 'dodgerblue', linewidth=2)[0], # ADDED
        }
        
        # --- Axis Titles and Setup ---
        ax_reward.set_title('Total Reward per Episode', fontweight='bold')
        ax_reward.set_xlabel('Episode')
        ax_reward.set_ylabel('Total Reward')
        ax_reward.grid(True, alpha=0.3)
        
        ax_loss.set_title('Losses', fontweight='bold')
        ax_loss.set_xlabel('Episode')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        ax_entropy_metric.set_title('Training Entropy (A2C)', fontweight='bold')
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
        
        ax_density.set_title('Grid Density', fontweight='bold')
        ax_density.set_xlabel('Episode')
        ax_density.set_ylabel('Density')
        ax_density.grid(True, alpha=0.3)
        
        ax_policy_entropy.set_title('Policy Entropy (H(π))', fontweight='bold')
        ax_policy_entropy.set_xlabel('Episode')
        ax_policy_entropy.set_ylabel('H(π)')
        ax_policy_entropy.grid(True, alpha=0.3)
        
        ax_avg_value.set_title('Average State Value (V(s))', fontweight='bold') # ADDED
        ax_avg_value.set_xlabel('Episode')
        ax_avg_value.set_ylabel('Avg V(s)')
        ax_avg_value.grid(True, alpha=0.3)

        plt.pause(0.1)

    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0.0
        ep_metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'advantage': 0.0,
            'entropy': 0.0,
            'td_error': 0.0,
            'density': 0.0,
            'policy_entropy': 0.0,
            'state_value': 0.0, 
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
            
            # Calculate policy entropy and value for current state
            action_probs_current, current_value = agent.get_action_probs_and_value(state)
            # Clip probabilities to avoid log(0)
            action_probs_current = np.clip(action_probs_current, 1e-10, 1.0) 
            policy_entropy = -np.sum(action_probs_current * np.log(action_probs_current))

            state = next_state
            episode_reward += float(reward)
            
            # Accumulate metrics
            ep_metrics['actor_loss'] += float(metrics['actor_loss'].numpy())
            ep_metrics['critic_loss'] += float(metrics['critic_loss'].numpy())
            ep_metrics['advantage'] += float(metrics['advantage'].numpy())
            ep_metrics['entropy'] += float(metrics['entropy'].numpy())
            ep_metrics['td_error'] += float(metrics['td_error'].numpy())
            ep_metrics['density'] += np.mean(env.ca_grid)
            ep_metrics['policy_entropy'] += policy_entropy
            ep_metrics['state_value'] += current_value 
            print(episode)

            pbar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Entropy': f'{ep_metrics["entropy"] / (step + 1):.3f}',
                'Value': f'{current_value:.2f}' 
            })

        # Store episode metrics
        total_rewards.append(episode_reward) # This corrects the total reward plot
        actor_losses.append(ep_metrics['actor_loss'] / args.steps)
        critic_losses.append(ep_metrics['critic_loss'] / args.steps)
        advantages.append(ep_metrics['advantage'] / args.steps)
        entropies.append(ep_metrics['entropy'] / args.steps)
        td_errors.append(ep_metrics['td_error'] / args.steps)
        density_values.append(float(ep_metrics['density'] / args.steps))
        policy_entropies.append(float(ep_metrics['policy_entropy'] / args.steps))
        avg_state_values.append(float(ep_metrics['state_value'] / args.steps)) # ADDED
        
        # Update live plot (modify mod frequency with print frequency desire)
        if args.live_plot and (episode + 1) % 5 == 0:
            episodes_range = range(1, episode + 2)
            
            # Update grid and agent position
            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            
            # Get current action probs and value (using the last state)
            action_probs, current_value = agent.get_action_probs_and_value(state)
            
            # Update action probability bars
            for bar, prob in zip(bars, action_probs):
                bar.set_height(prob)
            
            # Update value display (NEW AXIS)
            value_text.set_text(f'V(s) = {current_value:.3f}')
            
            # Update all metric lines
            lines['reward'].set_data(episodes_range, total_rewards)
            lines['actor_loss'].set_data(episodes_range, actor_losses)
            lines['critic_loss'].set_data(episodes_range, critic_losses)
            lines['entropy'].set_data(episodes_range, entropies)
            lines['td_error'].set_data(episodes_range, td_errors)
            lines['advantage'].set_data(episodes_range, advantages)
            lines['density'].set_data(episodes_range, density_values)
            lines['policy_entropy'].set_data(episodes_range, policy_entropies)
            lines['avg_value'].set_data(episodes_range, avg_state_values) # ADDED
            
            # Rescale all axes
            for ax in [ax_reward, ax_loss, ax_entropy_metric, ax_td, 
                       ax_advantage, ax_density, ax_policy_entropy, 
                       ax_avg_value]: # UPDATED list of axes
                ax.relim()
                ax.autoscale_view()
            
            fig.canvas.draw()
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
        'density_values': density_values,
        'policy_entropies': policy_entropies,
        'avg_state_values': avg_state_values,
    }
    
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_data, f)
    print("Training metrics saved to training_metrics.json")

    # Final plot (3x3 grid now)
    if args.live_plot:
        plt.ioff()
    
    plt.figure(figsize=(15, 12)) # Adjusted size for 3x3 layout
    
    # 1. Total Reward
    plt.subplot(3, 3, 1)
    plt.plot(total_rewards, linewidth=2)
    plt.title('Total Reward per Episode', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # 2. Losses
    plt.subplot(3, 3, 2)
    plt.plot(actor_losses, label='Actor Loss', linewidth=2)
    plt.plot(critic_losses, label='Critic Loss', linewidth=2)
    plt.title('Average Loss per Episode', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Average Advantage
    plt.subplot(3, 3, 3)
    plt.plot(advantages, linewidth=2, color='purple')
    plt.title('Average Advantage', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Advantage')
    plt.grid(True, alpha=0.3)
    
    # 4. Training Entropy
    plt.subplot(3, 3, 4)
    plt.plot(entropies, linewidth=2, color='orange')
    plt.title('Training Entropy (A2C)', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.grid(True, alpha=0.3)
    
    # 5. TD Error
    plt.subplot(3, 3, 5)
    plt.plot(td_errors, linewidth=2, color='brown')
    plt.title('TD Error', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('|TD Error|')
    plt.grid(True, alpha=0.3)
    
    # 6. Grid Density (Replaced Alive Cells)
    # plt.subplot(3, 3, 6)
    # plt.plot(density_values, linewidth=2, color='darkgreen')
    # plt.title('Grid Density', fontweight='bold')
    # plt.xlabel('Episode')
    # plt.ylabel('Density')
    # plt.grid(True, alpha=0.3)
    
    # 7. Policy Entropy (H(π))
    plt.subplot(3, 3, 7)
    plt.plot(policy_entropies, linewidth=2, color='crimson')
    plt.title('Policy Entropy (H(π))', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('H(π)')
    plt.grid(True, alpha=0.3)

    # 8. Average State Value (V(s)) (NEW)
    plt.subplot(3, 3, 8)
    plt.plot(avg_state_values, linewidth=2, color='dodgerblue')
    plt.title('Average State Value (V(s))', fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Avg V(s)')
    plt.grid(True, alpha=0.3)
    
    # 9. Empty space for a cleaner layout
    plt.subplot(3, 3, 9).axis('off')
    
    plt.tight_layout()
    plt.savefig('training_results_updated.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print("Final plot saved to training_results_updated.png")

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
    
    # plt.show()
    # plt.pause(0.001)
    plt.show(block=True)

def run_demo_auto(args):
    """Agent-controlled autonomous demo with agent-square, target, probs, value, and metrics."""
    import matplotlib as mpl
    mpl.rcParams['keymap.fullscreen'] = []

    print("--- Running Autonomous Demo ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)

    # Require weights for autonomous demo
    if not getattr(args, 'weights', None) or not os.path.exists(args.weights):
        raise FileNotFoundError(f"Autonomous demo requires a valid weights file. Provide --weights <file> (found: {getattr(args,'weights',None)})")

    agent = ActorCriticAgent(
        state_shape=(args.grid_size, args.grid_size, 2),
        num_actions=env.num_actions
    )
    agent.model.load_weights(args.weights)
    state = env.reset()

    # Layout (similar to original): if pattern reward show target axis
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

    # Main grid + patches
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

    # Action probs bars
    action_labels = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
    action_probs, state_value = agent.get_action_probs_and_value(state)
    bars = ax_probs.bar(range(env.num_actions), action_probs, color='steelblue', alpha=0.8)
    ax_probs.set_ylim([0, 1])
    ax_probs.set_xticks(range(env.num_actions))
    ax_probs.set_xticklabels(action_labels, fontsize=8)
    ax_probs.set_ylabel('Probability')
    ax_probs.set_title('Action Distribution')
    ax_probs.grid(axis='y', alpha=0.25)

    # Value display
    value_text = ax_value.text(0.5, 0.5, f'V(s) = {state_value:.3f}',
                               ha='center', va='center', fontsize=18,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_value.set_xlim([0, 1])
    ax_value.set_ylim([0, 1])
    ax_value.set_title('State Value (Critic)')
    ax_value.axis('off')

    # Metrics panel
    metrics_text = ax_metrics.text(0.05, 0.95, '', va='top', fontsize=10, family='monospace')
    ax_metrics.set_title('Metrics')
    ax_metrics.axis('off')

    # running stats
    value_history = deque(maxlen=200)
    entropy_history = deque(maxlen=200)

    def update(frame):
        nonlocal state
        action = agent.select_action(state)
        # get metrics for current state (before stepping)
        action_probs, state_value = agent.get_action_probs_and_value(state)
        clipped_probs = np.clip(action_probs, 1e-10, 1.0)
        entropy = -np.sum(clipped_probs * np.log(clipped_probs))
        value_history.append(state_value)
        entropy_history.append(entropy)

        # step environment
        next_state, reward, _, _ = env.step(action)
        state = next_state

        # update grid & patches
        grid_img.set_data(env.ca_grid)
        agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))

        if env.has_target:
            target_patch.set_xy((env.target_x - 0.5, env.target_y - 0.5))
            target_patch.set_visible(True)
        else:
            target_patch.set_visible(False)

        # update bars and highlight selected action
        for i, (bar, p) in enumerate(zip(bars, action_probs)):
            bar.set_height(p)
            bar.set_color('coral' if i == action else 'steelblue')

        # update value text
        value_text.set_text(f'V(s) = {state_value:.3f}')

        # update metrics block
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

        # return artists for FuncAnimation (makes update more robust)
        return [grid_img, agent_patch, target_patch] + list(bars)

    ani = animation.FuncAnimation(fig, update, frames=args.steps, interval=100, repeat=False, blit=False)
    print("\nAgent demo running (close window to stop).")
    plt.show(block=True)

def run_demo_manual(args):
    """Manual demo: shows agent/user square, target, probs, and metrics. r=record, p=save, q=quit."""
    import matplotlib as mpl
    mpl.rcParams['keymap.fullscreen'] = []

    print("--- Running Manual Demo ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)

    # Optional: show agent-based probabilities if user passed weights
    agent = None
    if getattr(args, 'weights', None) and os.path.exists(args.weights):
        agent = ActorCriticAgent(
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

    # Layout: mimic demo layout so user sees the same panels
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

    # Recording storage
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

        # Toggle recording
        if key == 'r':
            recording = not recording
            print(f"Recording {'started' if recording else 'stopped'}.")
            return

        # Save recorded data while window open
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

        # Quit
        if key == 'q' or key == 'escape':
            plt.close()
            return

        # Determine action from arrow / space / hex
        if key in key_map:
            action = key_map[key]
        elif key in hex_keys:
            action = 5 + hex_keys.index(key)
        else:
            # unknown key -> ignore
            return

        # log pre-action state if recording
        if recording:
            states_logged.append(np.squeeze(state, axis=0).copy())

        next_state, reward, _, _ = env.step(action)

        if recording:
            actions_logged.append(action)
            rewards_logged.append(reward)
            next_states_logged.append(np.squeeze(next_state, axis=0).copy())
            ts_logged.append(time.time())

        # update state pointer
        state = next_state
        step_counter += 1

        # update visuals
        grid_img.set_data(env.ca_grid)
        agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))

        if env.has_target:
            target_patch.set_xy((env.target_x - 0.5, env.target_y - 0.5))
            target_patch.set_visible(True)
        else:
            target_patch.set_visible(False)

        # If we have a loaded agent, show its probs for the *current* state,
        # otherwise show uniform but highlight user action on the bars.
        if agent is not None:
            action_probs, state_value = agent.get_action_probs_and_value(state)
        else:
            action_probs = np.ones(env.num_actions) / env.num_actions
            state_value = 0.0

        for i, (bar, p) in enumerate(zip(bars, action_probs)):
            bar.set_height(p)
            bar.set_color('coral' if i == action else 'steelblue')

        # Build metrics string
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

    # block until user closes window
    plt.show(block=True)

    # After window closed, auto-save recorded data if any (if user didn't press p)
    if len(actions_logged) > 0 and not os.path.exists(f'manual_data_{int(time.time())}.npz'):
        fname = f'manual_data_{int(time.time())}.npz'
        np.savez_compressed(fname,
                            states=np.array(states_logged),
                            actions=np.array(actions_logged),
                            rewards=np.array(rewards_logged),
                            next_states=np.array(next_states_logged),
                            timestamps=np.array(ts_logged))
        print(f"Manual demo finished. Saved {len(actions_logged)} samples to {fname}")
    elif len(actions_logged) == 0:
        print("Manual demo finished. No data recorded.")


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced RL in Cellular Automata Environment.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select mode: train, demo, or create_pattern')

    # --- Training Arguments ---
    train_parser = subparsers.add_parser('train', help='Run the training loop.')
    train_parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes.')
    train_parser.add_argument('--steps', type=int, default=200, help='Number of steps per episode.')
    train_parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate.')
    train_parser.add_argument('--entropy-coef', type=float, default=0.25, help='Entropy regularization coefficient.')
    train_parser.add_argument('--grid-size', type=int, default=12, help='Size of the CA grid.')
    train_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'], help='CA rule set.')
    train_parser.add_argument('--reward', type=str, default='entropy', choices=['entropy', 'maxwell_demon', 'target_practice', 'pattern'], help='Reward function.')
    train_parser.add_argument('--pattern-file', type=str, default=None, help='Load custom pattern from file for training.')
    train_parser.add_argument('--live-plot', action='store_true', help='Enable live plotting during training.')
    train_parser.add_argument('--pretrained-weights', type=str, default=None, help='Optional path to pretrained weights to load before RL training (e.g. from supervised pretrain).')

    # --- Supervised Learning Arguments ---
    sup_parser = subparsers.add_parser('supervised', help='Run supervised pretraining on recorded manual data.')
    sup_parser.add_argument('--data-file', type=str, required=True, help='Path to .npz file produced by manual recording (contains arrays states, actions).')
    sup_parser.add_argument('--epochs', type=int, default=10, help='Supervised training epochs.')
    sup_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for supervised training.')
    sup_parser.add_argument('--grid-size', type=int, default=12, help='CA grid size (must match dataset).')
    sup_parser.add_argument('--rules', type=str, default='conway', choices=['conway','seeds','maze'])
    sup_parser.add_argument('--reward', type=str, default='entropy', choices=['entropy','maxwell_demon','target_practice','pattern'])
    sup_parser.add_argument('--init-weights', type=str, default=None, help='Optional initial weights to load before supervised training.')
    sup_parser.add_argument('--save-weights', type=str, default='supervised_weights_final.weights.h5', help='Filename to save trained weights.')
    sup_parser.add_argument('--save-every', type=int, default=5, help='Save intermediate weights every N epochs.')
    sup_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for supervised training.')
    sup_parser.add_argument('--entropy-coef', type=float, default=0.0, help='Entropy regularization during supervised training.')
    # supervised learning should also pre-train critic, not just actor, so we have critic hyperparameters too
    sup_parser.add_argument('--gamma', type=float, default=0.99, help='Discount for critic target bootstrap.')
    sup_parser.add_argument('--value-coef', type=float, default=0.5, help='Weight of value loss in supervised pretraining.')

    # --- Demo Arguments ---
    demo_parser = subparsers.add_parser('demo', help='Run autonomous demo with trained agent.')
    demo_parser.add_argument('--weights', type=str, default='ca_agent_weights_final.weights.h5')
    demo_parser.add_argument('--steps', type=int, default=500)
    demo_parser.add_argument('--grid-size', type=int, default=12)
    demo_parser.add_argument('--rules', type=str, default='conway', choices=['conway','seeds','maze'])
    demo_parser.add_argument('--reward', type=str, default='entropy', choices=['entropy','maxwell_demon','target_practice','pattern'])
    demo_parser.add_argument('--pattern-file', type=str, default=None)

    # --- Manual Demo ---
    manual_parser = subparsers.add_parser('manual', help='Manual play + optional recording.')
    manual_parser.add_argument('--grid-size', type=int, default=12)
    manual_parser.add_argument('--rules', type=str, default='conway', choices=['conway','seeds','maze'])
    manual_parser.add_argument('--reward', type=str, default='entropy', choices=['entropy','maxwell_demon','target_practice','pattern'])
    manual_parser.add_argument('--pattern-file', type=str, default=None)
    manual_parser.add_argument('--steps', type=int, default=500)
    manual_parser.add_argument('--weights', type=str, default=None, help='(optional) weights file to visualize agent policy while playing')

    # --- Pattern Creator Arguments ---
    pattern_parser = subparsers.add_parser('create_pattern', help='Interactive pattern creation tool.')
    pattern_parser.add_argument('--grid-size', type=int, default=12, help='Size of the pattern grid.')


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