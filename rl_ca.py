import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import argparse
import logging
import os
import time

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --- Custom Keras Layer for Toroidal Padding ---
class ToroidalPadding(layers.Layer):
    """
    Custom Keras layer to apply toroidal (wrap-around) padding to a 4D tensor.
    This is essential for the convolutional network to correctly process the
    edges of the toroidal cellular automata grid.
    """
    def __init__(self, **kwargs):
        super(ToroidalPadding, self).__init__(**kwargs)

    def call(self, inputs):
        # input dims = [batch size, height, width, channels]
        # Pad top and bottom
        top_row = inputs[:, -1:, :, :]
        bottom_row = inputs[:, :1, :, :]
        vertical_padded = tf.concat([top_row, inputs, bottom_row], axis=1)

        # Pad left and right
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
    Handles the grid state, agent position, CA rules, and reward calculations.
    It provides a Gym-like interface with reset() and step() methods.
    """
    def __init__(self, grid_size=12, initial_density=0.4, rules_name='conway', reward_type='entropy', target_pattern=None, pattern_steps=50):
        self.grid_size = grid_size
        self.initial_density = initial_density
        self.rules_name = rules_name
        self.reward_type = reward_type
        self.pattern_steps = pattern_steps

        self.ca_rules = {
            'conway': {'birth': [3], 'survive': [2, 3]},
            'seeds': {'birth': [2], 'survive': []},
            'maze': {'birth': [3], 'survive': [1, 2, 3, 4, 5]}
        }
        self.rules = self.ca_rules[self.rules_name]

        self.actions = ['up', 'down', 'left', 'right', 'do_nothing'] + [f'write_{i:04b}' for i in range(16)]
        self.num_actions = len(self.actions)

        # For pattern generation mode
        if self.reward_type == 'pattern':
            if target_pattern is not None:
                self.target_pattern = target_pattern
            else:
                self.target_pattern = self._generate_target_pattern()
        else:
            self.target_pattern = None

        self.reset()

    def _generate_target_pattern(self):
        """Generates a target pattern by executing random actions from a blank grid."""
        print(f"\nGenerating target pattern with {self.pattern_steps} random actions...")
        
        # Create a temporary environment state
        temp_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        temp_agent_x = self.grid_size // 2
        temp_agent_y = self.grid_size // 2
        
        action_sequence = []
        
        for _ in range(self.pattern_steps):
            # Sample a random action
            action = np.random.randint(0, self.num_actions)
            action_sequence.append(action)
            
            # Execute the action
            if action == 0:  # up
                temp_agent_y = (temp_agent_y - 1 + self.grid_size) % self.grid_size
            elif action == 1:  # down
                temp_agent_y = (temp_agent_y + 1) % self.grid_size
            elif action == 2:  # left
                temp_agent_x = (temp_agent_x - 1 + self.grid_size) % self.grid_size
            elif action == 3:  # right
                temp_agent_x = (temp_agent_x + 1) % self.grid_size
            elif action == 4:  # do nothing
                pass
            elif action >= 5:  # write pattern
                pattern_index = action - 5
                bits = [(pattern_index >> 3) & 1, (pattern_index >> 2) & 1, 
                       (pattern_index >> 1) & 1, pattern_index & 1]
                write_pattern = np.array(bits).reshape(2, 2)
                
                # Write pattern
                for i in range(2):
                    for j in range(2):
                        y = (temp_agent_y + i - 1 + self.grid_size) % self.grid_size
                        x = (temp_agent_x + j - 1 + self.grid_size) % self.grid_size
                        temp_grid[y, x] = write_pattern[i, j]
            
            # Update CA
            temp_grid = self._apply_ca_rules(temp_grid)
        
        print(f"Target pattern generated. Action sequence: {[self.actions[a] for a in action_sequence]}")
        print(f"Target pattern density: {np.mean(temp_grid):.3f}")
        
        return temp_grid.copy()

    def _apply_ca_rules(self, grid):
        """Applies CA rules to a grid and returns the new grid."""
        new_grid = grid.copy()
        neighbor_counts = np.zeros_like(grid)
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbor_counts += np.roll(np.roll(grid, i, axis=0), j, axis=1)

        birth_mask = np.isin(neighbor_counts, self.rules['birth']) & (grid == 0)
        survive_mask = np.isin(neighbor_counts, self.rules['survive']) & (grid == 1)

        new_grid.fill(0)
        new_grid[birth_mask | survive_mask] = 1
        
        return new_grid

    def reset(self):
        """Resets the environment to an initial state."""
        if self.reward_type == 'target_practice':
            self.ca_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        elif self.reward_type == 'pattern':
            # Always start with blank grid for pattern generation task
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
        """
        Constructs the state tensor (grid + agent position channels).
        Returns a tensor of shape (1, grid_size, grid_size, 2).
        """
        agent_pos_channel = np.zeros_like(self.ca_grid, dtype=np.int8)
        
        # Mark the 2x2 area around the agent's top-left corner
        for i in range(-1, 1):
             for j in range(-1, 1):
                y = (self.agent_y + i) % self.grid_size
                x = (self.agent_x + j) % self.grid_size
                agent_pos_channel[y,x] = 1

        state = np.stack([self.ca_grid, agent_pos_channel], axis=-1)
        return np.expand_dims(state, axis=0).astype(np.float32)

    def step(self, action):
        """
        Executes one time step in the environment.
        Returns: next_state, reward, done, info
        """
        reward = 0
        current_pattern = None

        # --- 1. Execute Agent Action ---
        if 0 <= action <= 3: # Move actions
            if action == 0: self.agent_y = (self.agent_y - 1 + self.grid_size) % self.grid_size
            elif action == 1: self.agent_y = (self.agent_y + 1) % self.grid_size
            elif action == 2: self.agent_x = (self.agent_x - 1 + self.grid_size) % self.grid_size
            elif action == 3: self.agent_x = (self.agent_x + 1) % self.grid_size
        elif action == 4: # Do nothing
            pass
        elif action >= 5: # Write pattern
            pattern_index = action - 5
            bits = [(pattern_index >> 3) & 1, (pattern_index >> 2) & 1, (pattern_index >> 1) & 1, pattern_index & 1]
            current_pattern = np.array(bits).reshape(2, 2)

        # --- 2. Update Cellular Automata ---
        self._update_ca(current_pattern)
        
        # --- 3. Calculate Reward ---
        if self.reward_type == 'entropy':
            new_entropy = self._calculate_entropy()
            reward = -new_entropy
        elif self.reward_type == 'maxwell_demon':
            reward = self._calculate_separation_reward()
        elif self.reward_type == 'target_practice':
            reward = -0.1  # Small penalty for each step
            if self._check_target_destroyed():
                reward += 100
                self.ca_grid.fill(0) # Clear grid
                self._spawn_target()
        elif self.reward_type == 'pattern':
            binary_cross_entropy = self._calculate_pattern_bce()
            reward = -binary_cross_entropy
            # Optional: Add bonus for getting very close
            if binary_cross_entropy < 0.01:
                reward += 10  # Bonus reward for matching target

        # In this environment, 'done' is always False as episodes have fixed length
        done = False
        return self._get_state(), reward, done, {}

    def _update_ca(self, write_pattern=None):
        """Updates the CA grid for one step based on the rules."""
        new_grid = self.ca_grid.copy()
        
        # Convolve with a 3x3 kernel to get neighbor counts (update to use kernel)
        # kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_counts = np.zeros_like(self.ca_grid)
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbor_counts += np.roll(np.roll(self.ca_grid, i, axis=0), j, axis=1)

        birth_mask = np.isin(neighbor_counts, self.rules['birth']) & (self.ca_grid == 0)
        survive_mask = np.isin(neighbor_counts, self.rules['survive']) & (self.ca_grid == 1)

        new_grid.fill(0)
        new_grid[birth_mask | survive_mask] = 1
        
        self.ca_grid = new_grid

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
        
        # Flatten both grids
        current = self.ca_grid.flatten().astype(np.float32)
        target = self.target_pattern.flatten().astype(np.float32)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        current = np.clip(current, epsilon, 1 - epsilon)
        
        # Binary cross-entropy formula: -[y*log(p) + (1-y)*log(1-p)]
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

# --- Actor-Critic Agent ---
class ActorCriticAgent:
    """
    The Actor-Critic agent. Contains the policy (actor) and value (critic) networks,
    and handles action selection and training.
    """
    def __init__(self, state_shape, num_actions, learning_rate=0.0001, gamma=0.95):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        self._build_networks()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.huber_loss = keras.losses.Huber()

    def _build_networks(self):
        # Input layer expects the raw state, padding is applied inside the model
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
        # Use tf.random.categorical to sample an action
        action = tf.random.categorical(logits, 1)[0, 0].numpy()
        return action

    @tf.function
    def train_step(self, state, action, reward, next_state, done):
        """Performs a single training step for both actor and critic."""
        with tf.GradientTape() as tape:
            # Get model outputs for current and next states
            action_logits, state_value = self.model(state)
            _, next_state_value = self.model(next_state)

            # Critic loss calculation
            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            td_target = reward + self.gamma * next_state_value * (1 - float(done))
            advantage = td_target - state_value
            critic_loss = self.huber_loss(tf.expand_dims(td_target, 0), tf.expand_dims(state_value, 0))

            # Actor loss calculation (policy gradient)
            # The 'action' tensor (from numpy) defaults to int64, while tf.range defaults to int32.
            # tf.stack requires all input tensors to have the same dtype. We cast 'action' to int32 to resolve this.
            action_indices = tf.stack([tf.range(state.shape[0], dtype=tf.int32), tf.cast(action, tf.int32)], axis=1)
            log_probs = tf.nn.log_softmax(action_logits)
            action_log_probs = tf.gather_nd(log_probs, action_indices)
            
            # Stop gradient to treat advantage as a constant
            actor_loss = -action_log_probs * tf.stop_gradient(advantage)

            # Total loss (you can weight these if needed)
            total_loss = actor_loss + critic_loss

        # Compute and apply gradients
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return actor_loss, critic_loss

def train_agent(args):
    """Main training loop."""
    print("--- Starting Training ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward, 
                pattern_steps=args.pattern_steps)
    agent = ActorCriticAgent(
        state_shape=(args.grid_size, args.grid_size, 2),
        num_actions=env.num_actions,
        learning_rate=args.lr
    )

    total_rewards = []
    actor_losses = []
    critic_losses = []

    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0.0
        ep_actor_loss, ep_critic_loss = 0.0, 0.0

        pbar = tqdm(range(args.steps), desc=f"Episode {episode + 1}/{args.episodes}")
        for step in pbar:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            actor_loss, critic_loss = agent.train_step(
                state,
                np.array([action]),
                np.array([reward], dtype=np.float32),
                next_state,
                np.array([done])
            )

            state = next_state
            episode_reward += float(reward)
            ep_actor_loss += float(actor_loss.numpy())
            ep_critic_loss += float(critic_loss.numpy())

            pbar.set_postfix({
                'Reward': f'{episode_reward:.2f}',
                'Actor Loss': f'{ep_actor_loss / (step + 1):.3f}',
                'Critic Loss': f'{ep_critic_loss / (step + 1):.3f}'
            })

        total_rewards.append(episode_reward)
        actor_losses.append(ep_actor_loss / args.steps)
        critic_losses.append(ep_critic_loss / args.steps)

        # Save model weights periodically
        if (episode + 1) % 25 == 0:
            agent.model.save_weights(f'ca_agent_weights_{episode+1}.weights.h5')
            print(f"\nSaved weights at episode {episode+1}")

    agent.model.save_weights('ca_agent_weights_final.weights.h5')
    print("\n--- Training Complete ---")
    print(f"Final weights saved to ca_agent_weights_final.weights.h5")

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(total_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(actor_losses, label='Actor Loss')
    plt.plot(critic_losses, label='Critic Loss')
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_demo(args):
    """Runs a visualization of the environment with a trained agent or manual control."""
    import matplotlib as mpl

    # Disable default fullscreen binding for "f"
    mpl.rcParams['keymap.fullscreen'] = []

    print("--- Starting Demo ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, reward_type=args.reward,
                pattern_steps=args.pattern_steps)
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
    
    # Setup figure with target pattern display for 'pattern' mode
    if env.reward_type == 'pattern':
        fig, (ax_main, ax_target) = plt.subplots(1, 2, figsize=(12, 5))
        ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1)
        ax_target.set_title('Target Pattern')
        ax_target.set_xticks([])
        ax_target.set_yticks([])
    else:
        fig, ax_main = plt.subplots()
    
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

    current_action = 4  # do nothing
    step_requested = {'flag': False}

    key_map = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3,
        ' ': 4,  # do nothing
    }
    hex_keys = list('0123456789abcdef')  # 16 patterns

    def on_press(event):
        if event.key is None:
            return
        key = event.key.lower()
        nonlocal current_action

        if key in key_map:
            current_action = key_map[key]
            step_requested['flag'] = True
            print(f"Manual action set to: {env.actions[current_action]}")
        elif key in hex_keys:
            pattern_index = hex_keys.index(key)
            current_action = 5 + pattern_index
            step_requested['flag'] = True
            print(f"Manual write pattern: {pattern_index:04b} (action {env.actions[current_action]})")

    fig.canvas.mpl_connect('key_press_event', on_press)

    # --- Legend (ensure it's visible) ---
    legend_text = (
        "Controls:\n"
        "Arrows = Move\n"
        "Space  = Do nothing\n"
        "0–f    = Write 2x2 pattern\n\n"
        "Pattern bit order:\n"
        "top-left, top-right,\n"
        "bottom-left, bottom-right"
    )
    
    if env.reward_type == 'pattern':
        fig.subplots_adjust(right=0.85)
        fig.text(0.87, 0.5, legend_text, va='center', fontsize=9, family='monospace')
    else:
        fig.subplots_adjust(right=0.7)
        fig.text(0.72, 0.5, legend_text, va='center', fontsize=9, family='monospace')

    if manual_mode:
        print("\nManual Control Enabled (single-step):")
        print("- Arrow Keys to move (one step per key press)")
        print("- Spacebar to pass (do nothing)")
        print("- Hex keys 0-9 and a-f to write 2x2 patterns (each hex value is the 4-bit pattern: top-left → top-right → bottom-left → bottom-right)")
        print("- Focus the plot window to register keys. Each keypress advances one step.\n")

        step = 0
        while step < args.steps:
            plt.pause(0.01)
            if not step_requested['flag']:
                continue
            action = current_action
            current_action = 4
            step_requested['flag'] = False

            next_state, reward, _, _ = env.step(action)
            state[:] = next_state
            state = next_state

            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))

            if env.has_target:
                target_patch.set_xy((env.target_x - 0.5, env.target_y - 0.5))
                target_patch.set_visible(True)
            else:
                target_patch.set_visible(False)

            if env.reward_type == 'pattern':
                bce = env._calculate_pattern_bce()
                title_text.set_text(f"Step: {step} | Mode: Manual | Action: {env.actions[action]} | BCE: {bce:.4f}")
            else:
                title_text.set_text(f"Step: {step} | Mode: Manual | Last Action: {env.actions[action]}")
            
            fig.canvas.draw_idle()
            step += 1

        print("Manual demo finished.")
        plt.show(block=True)
    else:
        def update(frame):
            nonlocal state
            action = agent.select_action(state)
            next_state, _, _, _ = env.step(action)
            state = next_state

            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            if env.has_target:
                target_patch.set_xy((env.target_x - 0.5, env.target_y - 0.5))
                target_patch.set_visible(True)
            else:
                target_patch.set_visible(False)

            if env.reward_type == 'pattern':
                bce = env._calculate_pattern_bce()
                title_text.set_text(f"Step: {frame} | Mode: Agent | Action: {env.actions[action]} | BCE: {bce:.4f}")
            else:
                title_text.set_text(f"Step: {frame} | Mode: Agent | Last Action: {env.actions[action]}")
            
            return grid_img, agent_patch, target_patch, title_text

        ani = animation.FuncAnimation(fig, update, frames=args.steps, interval=100, repeat=False, blit=False)
        print("\nAgent demo running (close window to stop).")
        plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reinforcement Learning in a Cellular Automata Environment.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select mode: train or demo')

    # --- Training Arguments ---
    train_parser = subparsers.add_parser('train', help='Run the training loop.')
    train_parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes.')
    train_parser.add_argument('--steps', type=int, default=200, help='Number of steps per episode.')
    train_parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the Adam optimizer.')
    train_parser.add_argument('--grid-size', type=int, default=12, help='Size of the CA grid (e.g., 12 for 12x12).')
    train_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'], help='CA rule set to use.')
    train_parser.add_argument('--reward', type=str, default='entropy', choices=['entropy', 'maxwell_demon', 'target_practice', 'pattern'], help='Reward function to use.')
    train_parser.add_argument('--pattern-steps', type=int, default=10, help='Number of random actions to generate target pattern (for pattern mode).')

    # --- Demo Arguments ---
    demo_parser = subparsers.add_parser('demo', help='Run a visual demonstration.')
    demo_parser.add_argument('--weights', type=str, default='ca_agent_weights_final.weights.h5', help='Path to saved model weights. If not found, starts in manual mode.')
    demo_parser.add_argument('--steps', type=int, default=500, help='Number of steps to run the demo for.')
    demo_parser.add_argument('--grid-size', type=int, default=12, help='Size of the CA grid.')
    demo_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'], help='CA rule set to use.')
    demo_parser.add_argument('--reward', type=str, default='entropy', choices=['entropy', 'maxwell_demon', 'target_practice', 'pattern'], help='Reward function (affects environment setup).')
    demo_parser.add_argument('--pattern-steps', type=int, default=10, help='Number of random actions to generate target pattern (for pattern mode).')

    args = parser.parse_args()

    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'demo':
        run_demo(args)