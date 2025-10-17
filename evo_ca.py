#!/usr/bin/env python3
"""
evo_ca.py

Evolutionary Algorithm for Cellular Automata Pattern Matching
Uses genetic algorithms to evolve sequences of actions that create target patterns.

Author: Enhanced by Claude with Evolutionary Algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.signal import convolve2d
from tqdm import tqdm
import argparse
import os
import time
from collections import deque
import json

plt.ion()

# --- Cellular Automata Environment ---
class CAEnv:
    """Represents the Cellular Automata (CA) environment."""
    def __init__(self, grid_size=12, initial_density=0.4, rules_name='conway',
                 reward_type='pattern', target_pattern=None, max_steps=10):
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

        self.ca_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

        if target_pattern is not None:
            self.target_pattern = target_pattern
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
        self.ca_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.agent_x = self.grid_size // 2
        self.agent_y = self.grid_size // 2
        self.current_step = 0
        return self.ca_grid.copy()

    def step(self, action):
        """Executes one time step in the environment."""
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

        # Update Cellular Automata
        self._update_ca_fast(current_pattern)

        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        return self.ca_grid.copy(), done

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

    def calculate_fitness(self, final_grid):
        """Calculate fitness based on match with target pattern."""
        if self.target_pattern is None:
            return 0.0

        # Perfect match bonus
        match_fraction = np.mean(final_grid == self.target_pattern)
        fitness = match_fraction * 100

        # Extra bonus for perfect match
        if match_fraction == 1.0:
            fitness += 100

        return fitness

    def load_pattern(self, filename):
        """Load pattern from file."""
        if os.path.exists(filename):
            self.target_pattern = np.load(filename)
            print(f"Pattern loaded from {filename}")
            return True
        return False

    def save_pattern(self, filename):
        """Save current grid as target pattern."""
        if filename:
            np.save(filename, self.ca_grid)
            print(f"Pattern saved to {filename}")


# --- Evolutionary Algorithm ---
class EvolutionaryOptimizer:
    """Evolutionary algorithm to optimize action sequences."""

    def __init__(self, env, steps=10, population_size=100,
                 elite_fraction=0.2, mutation_rate=0.1):
        self.env = env
        self.steps = steps
        self.population_size = population_size
        self.elite_size = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate
        self.num_actions = env.num_actions

        # Initialize random population
        self.population = [self._random_sequence() for _ in range(population_size)]
        self.fitness_scores = np.zeros(population_size)

        # Track best solutions
        self.best_sequence = None
        self.best_fitness = -float('inf')
        self.best_history = []
        self.overall_best = []  # list of (sequence, fitness)

        # Statistics
        self.generation = 0
        self.avg_fitness_history = []
        self.max_fitness_history = []
        self.diversity_history = []

    def _random_sequence(self):
        """Generate a random action sequence."""
        return np.random.randint(0, self.num_actions, size=self.steps)

    def evaluate_sequence(self, sequence):
        """Evaluate fitness of an action sequence."""
        self.env.reset()

        for action in sequence:
            _, done = self.env.step(action)
            if done:
                break

        final_grid = self.env.ca_grid.copy()
        fitness = self.env.calculate_fitness(final_grid)

        return fitness, final_grid

    # def evaluate_population(self):
    #     """Evaluate all sequences in the population."""
    #     for i in range(self.population_size):
    #         self.fitness_scores[i], _ = self.evaluate_sequence(self.population[i])

    #         # Track best solution           
    #         if self.fitness_scores[i] > self.best_fitness:
    #             self.best_fitness = self.fitness_scores[i]
    #             self.best_sequence = self.population[i].copy()
    #             # Track in overall leaderboard
    #             self.overall_best.append((self.best_sequence.copy(), self.best_fitness))
    #             self.overall_best = sorted(self.overall_best, key=lambda x: x[1], reverse=True)[:5]

    #     # Update statistics
    #     self.avg_fitness_history.append(np.mean(self.fitness_scores))
    #     self.max_fitness_history.append(np.max(self.fitness_scores))
    #     self.diversity_history.append(self._calculate_diversity())

    def evaluate_population(self):
        """Evaluate all sequences in the population."""
        for i in range(self.population_size):
            self.fitness_scores[i], _ = self.evaluate_sequence(self.population[i])

            # Track current generation best
            if self.fitness_scores[i] > self.best_fitness:
                self.best_fitness = self.fitness_scores[i]
                self.best_sequence = self.population[i].copy()

        # Update statistics
        self.avg_fitness_history.append(np.mean(self.fitness_scores))
        self.max_fitness_history.append(np.max(self.fitness_scores))
        self.diversity_history.append(self._calculate_diversity())

        # --- Update both leaderboards ---
        # Get top 5 of this generation
        gen_top5 = self.get_top_sequences(k=5)

        # Merge them into the overall leaderboard (bounded at 10 entries)
        all_candidates = self.overall_best + gen_top5
        self.overall_best = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:5]


    def _calculate_diversity(self):
        """Calculate population diversity (unique sequences)."""
        unique = len(set(tuple(seq) for seq in self.population))
        return unique / self.population_size

    def get_top_sequences(self, k=5):
        """Get top k sequences by fitness."""
        top_indices = np.argsort(self.fitness_scores)[-k:][::-1]
        return [(self.population[i].copy(), self.fitness_scores[i]) for i in top_indices]

    def select_parents(self):
        """Select elite individuals for breeding."""
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        return [self.population[i].copy() for i in elite_indices]

    def crossover(self, parent1, parent2):
        """Single-point crossover between two parents."""
        crossover_point = np.random.randint(1, self.steps)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child

    def mutate(self, sequence):
        """Mutate a sequence by randomly changing some actions."""
        mutated = sequence.copy()
        for i in range(self.steps):
            if np.random.random() < self.mutation_rate:
                mutated[i] = np.random.randint(0, self.num_actions)
        return mutated

    def evolve(self):
        """Perform one generation of evolution."""
        # Select elite parents
        parents = self.select_parents()

        # Generate new population
        new_population = parents.copy()  # Keep elites

        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(len(parents), size=2, replace=False)
            child = self.crossover(parents[parent1], parents[parent2])
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population[:self.population_size]
        self.generation += 1


# --- Visualization Helpers ---
def create_action_images(num_actions, size=20):
    """
    Pre-generates improved images for each possible action, ensuring they are not cut off.
    The 'do nothing' action is represented by an empty set symbol (∅).
    """
    action_images = {}

    # --- Arrows (0-3) ---
    # Create arrow inside a slightly smaller canvas and pad it to prevent cutoff
    inner_size = size - 4 # Use a larger margin for clarity
    img = np.zeros((inner_size, inner_size))
    mid = inner_size // 2
    
    # Up Arrow (0)
    head_size = inner_size // 3
    # Draw arrowhead
    for i in range(head_size):
        img[i, mid - i : mid + i + 1] = 1
    # Draw shaft
    img[head_size:, mid - 1 : mid + 2] = 1
    # Pad the smaller canvas to the final size
    action_images[0] = np.pad(img, 2, 'constant', constant_values=0)

    # Down Arrow (1)
    action_images[1] = np.flipud(action_images[0])

    # Left Arrow (2)
    action_images[2] = np.rot90(action_images[0], 1)

    # Right Arrow (3)
    action_images[3] = np.rot90(action_images[0], -1)

    # --- Wait/Do Nothing (4) -> Empty Set symbol ---
    img = np.zeros((size, size))
    center = size // 2
    radius = size // 3 + 1
    yy, xx = np.ogrid[-center:size-center, -center:size-center]
    dist_sq = xx*xx + yy*yy
    
    # Draw the circle (as a stroke)
    circle_mask = (dist_sq <= radius*radius) & (dist_sq >= (radius-2)**2)
    
    # Draw the diagonal slash (top-left to bottom-right)
    y_idx, x_idx = np.indices((size, size))
    # Equation for a line y=x through the center
    line_mask = abs(y_idx - x_idx) < 2
    
    # Combine the circle and the slash
    img[circle_mask | line_mask] = 1
    action_images[4] = img

    # --- Write Patterns (5 to 20) ---
    for i in range(16):
        action_id = i + 5
        bits = [(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1]
        pattern = np.array(bits).reshape(2, 2)
        
        # Scale up pattern and add a 1px border for visual separation
        cell_size = (size - 2) // 2
        pattern_img = np.kron(pattern, np.ones((cell_size, cell_size)))
        pattern_img = np.pad(pattern_img, 1, 'constant', constant_values=0)
        action_images[action_id] = pattern_img

    return action_images


def train_evolutionary(args):
    """Main evolutionary training loop."""
    print("--- Starting Evolutionary Training ---")

    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules,
                reward_type='pattern', max_steps=args.steps)

    # Load pattern
    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)
    else:
        print("Warning: No pattern file loaded. Using empty grid as target.")

    optimizer = EvolutionaryOptimizer(
        env=env,
        steps=args.steps,
        population_size=args.population_size,
        elite_fraction=args.elite_fraction,
        mutation_rate=args.mutation_rate
    )

    # Setup live plotting
    if args.live_plot is not None:
        action_images = create_action_images(optimizer.num_actions, size=20)

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 5, hspace=0.5, wspace=0.4)

        # Grid displays
        ax_best = fig.add_subplot(gs[0, 0])
        ax_target = fig.add_subplot(gs[0, 1])
        ax_current = fig.add_subplot(gs[0, 2])

        # Metrics
        ax_fitness = fig.add_subplot(gs[0, 3:])
        ax_diversity = fig.add_subplot(gs[1, 0])
        ax_actions = fig.add_subplot(gs[1, 1])
        ax_dist = fig.add_subplot(gs[1, 2])
        ax_fitness_dist = fig.add_subplot(gs[1, 3:])
        # ax_leaderboard = fig.add_subplot(gs[2, 0:2])
        ax_leaderboard_gen = fig.add_subplot(gs[2, 0])
        ax_leaderboard_all = fig.add_subplot(gs[2, 1])
        ax_info = fig.add_subplot(gs[2, 2:4])

        ax_seq_imgs = fig.add_subplot(gs[2, 4])

        fig.suptitle('Evolutionary Algorithm Progress', fontsize=16, fontweight='bold')

        # Initialize displays
        best_img = ax_best.imshow(np.zeros((env.grid_size, env.grid_size)),
                                  cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax_best.set_title('Best Solution', fontweight='bold')
        ax_best.set_xticks([])
        ax_best.set_yticks([])

        target_img = ax_target.imshow(env.target_pattern if env.target_pattern is not None else np.zeros((env.grid_size, env.grid_size)),
                                      cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax_target.set_title('Target Pattern', fontweight='bold')
        ax_target.set_xticks([])
        ax_target.set_yticks([])

        current_img = ax_current.imshow(np.zeros((env.grid_size, env.grid_size)),
                                       cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax_current.set_title(f"Generation Sample (Step {args.steps})", fontweight='bold')
        ax_current.set_xticks([])
        ax_current.set_yticks([])

        # Fitness plot
        line_max = ax_fitness.plot([], [], 'b-', linewidth=2, label='Max')[0]
        line_avg = ax_fitness.plot([], [], 'g-', linewidth=2, label='Avg')[0]
        ax_fitness.set_title('Fitness Progress', fontweight='bold')
        ax_fitness.set_xlabel('Generation')
        ax_fitness.set_ylabel('Fitness')
        ax_fitness.legend()
        ax_fitness.grid(True, alpha=0.3)

        # Diversity plot
        line_div = ax_diversity.plot([], [], 'orange', linewidth=2)[0]
        ax_diversity.set_title('Population Diversity', fontweight='bold')
        ax_diversity.set_xlabel('Generation')
        ax_diversity.set_ylabel('Unique Sequences %')
        ax_diversity.set_ylim([0, 1])
        ax_diversity.grid(True, alpha=0.3)

        # Action sequence labels and colors
        action_labels = ['↑', '↓', '←', '→', '∅'] + [f'{i:X}' for i in range(16)]
        action_colors = plt.cm.tab20(np.linspace(0, 1, len(action_labels)))

        # Fitness distribution
        ax_fitness_dist.set_title('Fitness Distribution', fontweight='bold')
        ax_fitness_dist.set_xlabel('Fitness')
        ax_fitness_dist.set_ylabel('Count')

        # Action distribution
        ax_dist.set_title('Action Usage', fontweight='bold')
        ax_dist.set_ylabel('Frequency')
        
        # Leaderboard
        # leaderboard_text = ax_leaderboard.text(0.05, 0.95, '', va='top', fontsize=10, family='monospace',
        #                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        # ax_leaderboard.set_title('Top 5 Sequences', fontweight='bold')
        # ax_leaderboard.axis('off')
        leaderboard_text_gen = ax_leaderboard_gen.text(0.05, 0.95, '', va='top', fontsize=10, family='monospace',
                                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax_leaderboard_gen.set_title('Top 5 (Current Generation)', fontweight='bold')
        ax_leaderboard_gen.axis('off')

        leaderboard_text_all = ax_leaderboard_all.text(0.05, 0.95, '', va='top', fontsize=10, family='monospace',
                                                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax_leaderboard_all.set_title('Top 5 (All Generations)', fontweight='bold')
        ax_leaderboard_all.axis('off')

        
        # New Action sequence visualization
        ax_seq_imgs.set_title('Best Sequence', fontweight='bold')
        ax_seq_imgs.axis('off')

        # Info text
        info_text = ax_info.text(0.05, 0.95, '', va='top', fontsize=10, family='monospace')
        ax_info.set_title('Statistics', fontweight='bold')
        ax_info.axis('off')

        plt.pause(0.1)

    # Training loop
    print(f"\nTraining for {args.generations} generations...")
    print(f"Population size: {args.population_size}, Sequence length: {args.steps}")
    print(f"Elite fraction: {args.elite_fraction}, Mutation rate: {args.mutation_rate}\n")

    for generation in tqdm(range(args.generations), desc="Evolution"):
        # Evaluate population
        optimizer.evaluate_population()

        # Update visualization
        if args.live_plot is not None and (generation % args.live_plot == 0 or generation == args.generations - 1):
            # Evaluate best sequence for display
            best_fitness, best_grid = optimizer.evaluate_sequence(optimizer.best_sequence)

            # Update grid displays
            best_img.set_data(best_grid)

            # Evaluate a random current member
            random_idx = np.random.randint(0, optimizer.population_size)
            _, current_grid = optimizer.evaluate_sequence(optimizer.population[random_idx])
            current_img.set_data(current_grid)

            # Update fitness plot
            gens = range(len(optimizer.max_fitness_history))
            line_max.set_data(gens, optimizer.max_fitness_history)
            line_avg.set_data(gens, optimizer.avg_fitness_history)
            ax_fitness.relim()
            ax_fitness.autoscale_view()

            # Update diversity plot
            line_div.set_data(gens, optimizer.diversity_history)
            ax_diversity.relim()
            ax_diversity.autoscale_view(scalex=True, scaley=False) # Keep Y fixed

            # Action sequence bar chart
            ax_actions.clear()
            colors = [action_colors[a] for a in optimizer.best_sequence]
            ax_actions.bar(range(args.steps), optimizer.best_sequence, color=colors)
            ax_actions.set_title('Best Action Sequence', fontweight='bold')
            ax_actions.set_xlabel('Step')
            ax_actions.set_ylabel('Action ID')
            ax_actions.set_ylim([-1, optimizer.num_actions])
            ax_actions.grid(axis='y', alpha=0.3)

            # New: Action sequence image visualization
            ax_seq_imgs.clear()
            ax_seq_imgs.set_title('Best Sequence', fontweight='bold')
            ax_seq_imgs.axis('off')
            best_seq = optimizer.best_sequence
            if best_seq is not None and len(best_seq) > 0:
                seq_img_list = [action_images[action] for action in best_seq]
                padding = np.zeros_like(seq_img_list[0][:, :2])
                padded_imgs = []
                for img in seq_img_list:
                    padded_imgs.append(img)
                    padded_imgs.append(padding)
                composite_img = np.hstack(padded_imgs[:-1])
                ax_seq_imgs.imshow(composite_img, cmap='binary', interpolation='nearest')
            else:
                ax_seq_imgs.text(0.5, 0.5, 'No sequence', ha='center', va='center')

            # Fitness distribution
            ax_fitness_dist.clear()
            ax_fitness_dist.hist(optimizer.fitness_scores, bins=20, color='steelblue', alpha=0.7)
            ax_fitness_dist.set_title('Fitness Distribution', fontweight='bold')
            ax_fitness_dist.set_xlabel('Fitness')
            ax_fitness_dist.set_ylabel('Count')
            ax_fitness_dist.axvline(optimizer.best_fitness, color='red', linestyle='--', linewidth=2, label='Best')
            ax_fitness_dist.legend()

            # Action usage distribution
            ax_dist.clear()
            all_actions = np.concatenate(optimizer.population)
            action_counts = np.bincount(all_actions, minlength=optimizer.num_actions)
            bars = ax_dist.bar(range(optimizer.num_actions), action_counts, color='steelblue', alpha=0.7)
            ax_dist.set_title('Action Usage', fontweight='bold')
            ax_dist.set_ylabel('Frequency')
            ax_dist.set_xticks(range(optimizer.num_actions))
            ax_dist.set_xticklabels(action_labels, fontsize=7, rotation=45)

            # Highlight best sequence actions
            best_action_counts = np.bincount(optimizer.best_sequence, minlength=optimizer.num_actions)
            for i, bar in enumerate(bars):
                if best_action_counts[i] > 0:
                    bar.set_color('coral')

            # Update leaderboard
            # top_sequences = optimizer.get_top_sequences(k=5)
            # leaderboard_str = "LEADERBOARD (Top 5):\n" + "=" * 40 + "\n"
            # for rank, (seq, fitness) in enumerate(top_sequences, 1):
            #     seq_str = ' '.join([action_labels[a] for a in seq])
            #     leaderboard_str += f"#{rank}: Fitness {fitness:.2f}\n"
            #     leaderboard_str += f"    {seq_str}\n"
            # leaderboard_text.set_text(leaderboard_str)

            # --- Update generation leaderboard ---
            top_sequences = optimizer.get_top_sequences(k=5)
            leaderboard_str_gen = "LEADERBOARD (Gen):\n" + "=" * 30 + "\n"
            for rank, (seq, fitness) in enumerate(top_sequences, 1):
                seq_str = ' '.join([action_labels[a] for a in seq])
                leaderboard_str_gen += f"#{rank}: {fitness:.2f}\n    {seq_str}\n"
            leaderboard_text_gen.set_text(leaderboard_str_gen)

            # --- Update overall leaderboard ---
            leaderboard_str_all = "LEADERBOARD (All):\n" + "=" * 30 + "\n"
            for rank, (seq, fitness) in enumerate(optimizer.overall_best, 1):
                seq_str = ' '.join([action_labels[a] for a in seq])
                leaderboard_str_all += f"#{rank}: {fitness:.2f}\n    {seq_str}\n"
            leaderboard_text_all.set_text(leaderboard_str_all)

            # Update info text
            info_str = (
                f"Generation: {generation + 1}/{args.generations}\n"
                f"Best Fitness: {optimizer.best_fitness:.2f}\n"
                f"Avg Fitness: {optimizer.avg_fitness_history[-1]:.2f}\n"
                f"Diversity: {optimizer.diversity_history[-1]:.2%}\n"
                f"Perfect Matches: {np.sum(optimizer.fitness_scores >= 200)}\n"
            )

            if generation == args.generations - 1:
                info_str += f"\nFinal Score: {optimizer.best_fitness:.2f}\n"

            info_text.set_text(info_str)

            fig.canvas.draw()
            plt.pause(0.01)

        # Evolve population
        if generation < args.generations - 1:
            optimizer.evolve()

        # Save checkpoint
        if (generation + 1) % args.save_freq == 0:
            checkpoint = {
                'generation': generation + 1,
                'best_sequence': optimizer.best_sequence.tolist(),
                'best_fitness': float(optimizer.best_fitness),
                'max_fitness_history': optimizer.max_fitness_history,
                'avg_fitness_history': optimizer.avg_fitness_history,
                'diversity_history': optimizer.diversity_history
            }
            with open(f'checkpoint_gen{generation+1}.json', 'w') as f:
                json.dump(checkpoint, f, indent=2)

    # Save final results
    print(f"\n--- Training Complete ---")
    print(f"Best Fitness: {optimizer.best_fitness:.2f}")
    print(f"Best Sequence: {optimizer.best_sequence}")

    final_results = {
        'best_sequence': optimizer.best_sequence.tolist(),
        'best_fitness': float(optimizer.best_fitness),
        'max_fitness_history': optimizer.max_fitness_history,
        'avg_fitness_history': optimizer.avg_fitness_history,
        'diversity_history': optimizer.diversity_history,
        'action_labels': action_labels
    }

    with open('evolutionary_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # Save best sequence
    np.save('best_sequence.npy', optimizer.best_sequence)

    if args.live_plot is not None:
        plt.ioff()
        plt.savefig('evolutionary_results.png', dpi=150, bbox_inches='tight')

    print("\nResults saved to evolutionary_results.json and best_sequence.npy")


def run_demo(args):
    """Demonstrate a saved action sequence."""
    print("--- Running Sequence Demo ---")

    if not os.path.exists(args.sequence_file):
        raise FileNotFoundError(f"Sequence file not found: {args.sequence_file}")

    sequence = np.load(args.sequence_file)
    print(f"Loaded sequence of length {len(sequence)}")

    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules,
                reward_type='pattern', max_steps=len(sequence))
    
    action_images = create_action_images(env.num_actions, size=20)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)

    # Setup visualization
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 1], hspace=0.4, wspace=0.3)

    ax_grid = fig.add_subplot(gs[0, 0])
    ax_target = fig.add_subplot(gs[0, 1])
    ax_actions = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])
    ax_seq_imgs = fig.add_subplot(gs[2, :])

    # Target pattern
    if env.target_pattern is not None:
        ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    ax_target.set_title('Target Pattern', fontweight='bold')
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    # Grid
    state = env.reset()
    grid_img = ax_grid.imshow(state, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                facecolor='none', edgecolor='cyan', linewidth=2)
    ax_grid.add_patch(agent_patch)
    title_text = ax_grid.set_title("Step 0", fontweight='bold')
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    # Action sequence bar chart
    action_labels = ['↑', '↓', '←', '→', '∅'] + [f'{i:X}' for i in range(16)]
    action_colors = plt.cm.tab20(np.linspace(0, 1, len(action_labels)))
    colors = [action_colors[a] for a in sequence]
    bars = ax_actions.bar(range(len(sequence)), sequence, color=colors)
    ax_actions.set_title('Action Sequence', fontweight='bold')
    ax_actions.set_xlabel('Step')
    ax_actions.set_ylabel('Action ID')
    ax_actions.grid(axis='y', alpha=0.3)

    # Info
    info_text = ax_info.text(0.05, 0.95, '', va='top', fontsize=11, family='monospace')
    ax_info.set_title('Info', fontweight='bold')
    ax_info.axis('off')

    # Action image sequence
    ax_seq_imgs.set_title('Action History', fontweight='bold')
    ax_seq_imgs.axis('off')

    def update(frame):
        # Reset environment at the start of each loop
        if frame == 0:
            env.reset()
            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            title_text.set_text("Step 0")

            # Reset bar colors
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
                bar.set_alpha(1.0)
            
            # Clear image sequence
            ax_seq_imgs.clear()
            ax_seq_imgs.set_title('Action History', fontweight='bold')
            ax_seq_imgs.axis('off')

            info_text.set_text("Step: 0\nStarting...")

        elif frame <= len(sequence):
            action = sequence[frame - 1]
            state, _ = env.step(action)

            grid_img.set_data(state)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            title_text.set_text(f"Step {frame}")

            # Highlight current action
            for i, bar in enumerate(bars):
                if i == frame - 1:
                    bar.set_color('red')
                    bar.set_alpha(1.0)
                elif i < frame - 1:
                    bar.set_color(colors[i])
                    bar.set_alpha(0.3)
                else:
                    bar.set_color(colors[i])
                    bar.set_alpha(1.0)

            # Update action image sequence
            current_sequence = sequence[:frame]
            if len(current_sequence) > 0:
                seq_img_list = [action_images[act] for act in current_sequence]
                padding = np.zeros_like(seq_img_list[0][:, :2])
                padded_imgs = [img for act_img in seq_img_list for img in (act_img, padding)][:-1]
                composite_img = np.hstack(padded_imgs)
                ax_seq_imgs.clear()
                ax_seq_imgs.set_title('Action History', fontweight='bold')
                ax_seq_imgs.axis('off')
                ax_seq_imgs.imshow(composite_img, cmap='binary', interpolation='nearest')


            fitness = env.calculate_fitness(state)
            match_pct = np.mean(state == env.target_pattern) if env.target_pattern is not None else 0

            info_str = (
                f"Step: {frame}/{len(sequence)}\n"
                f"Action: {action_labels[action]}\n"
                f"Fitness: {fitness:.2f}\n"
                f"Match: {match_pct:.1%}\n"
                f"Alive cells: {np.sum(state)}"
            )
            if frame == len(sequence):
                info_str += f"\n\nFinal Score: {fitness:.2f}"

            info_text.set_text(info_str)

        return [grid_img, agent_patch] + list(bars) + [info_text]

    # Add pause frames at the end before looping
    total_frames = len(sequence) + 1 + 20  # 0 (reset) + sequence steps + 20 pause frames

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                 interval=500, repeat=True, blit=False)
    plt.show(block=True)


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
    ax.set_xticks(np.arange(-.5, args.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, args.grid_size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)


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


def run_demo_manual(args):
    """Manual play mode with keyboard control and pattern saving."""
    import matplotlib as mpl
    mpl.rcParams['keymap.fullscreen'] = []

    print("--- Running Manual Play Mode ---")
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules,
                reward_type='pattern', max_steps=args.steps)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)

    state = env.reset()

    # Setup visualization
    if env.target_pattern is not None:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_target = fig.add_subplot(gs[:, 1])
        ax_actions = fig.add_subplot(gs[0, 2:])
        ax_metrics = fig.add_subplot(gs[1, 2:])
        ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        ax_target.set_title('Target Pattern', fontweight='bold')
        ax_target.set_xticks([])
        ax_target.set_yticks([])
    else:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_actions = fig.add_subplot(gs[0, 1:])
        ax_metrics = fig.add_subplot(gs[1, 1:])

    grid_img = ax_main.imshow(env.ca_grid, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                facecolor='none', edgecolor='cyan', linewidth=2)
    ax_main.add_patch(agent_patch)
    title_text = ax_main.set_title("Step: 0 | Manual Mode", fontweight='bold')
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    action_labels = ['↑', '↓', '←', '→', '∅'] + [f'{i:X}' for i in range(16)]
    action_history = deque(maxlen=args.steps)

    bars = ax_actions.bar(range(len(action_labels)), np.zeros(len(action_labels)),
                          color='steelblue', alpha=0.7)
    ax_actions.set_ylim([0, 1])
    ax_actions.set_xticks(range(len(action_labels)))
    ax_actions.set_xticklabels(action_labels, fontsize=8)
    ax_actions.set_ylabel('Usage')
    ax_actions.set_title('Action History', fontweight='bold')
    ax_actions.grid(axis='y', alpha=0.3)

    metrics_text = ax_metrics.text(0.05, 0.95, '', va='top', fontsize=11, family='monospace')
    ax_metrics.set_title('Statistics', fontweight='bold')
    ax_metrics.axis('off')

    step_counter = [0]
    key_map = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3,
        ' ': 4,
    }
    hex_keys = list('0123456789abcdef')

    def on_key(event):
        if event.key is None:
            return
        key = event.key.lower()

        if key == 'q' or key == 'escape':
            plt.close()
            return
        
        if step_counter[0] >= args.steps and key not in ['c', 'q', 'escape']:
            metrics_text.set_text(f"{metrics_text.get_text().split('Controls:')[0]}\nMax steps reached!\nPress 'C' to clear or 'Q' to quit.")
            fig.canvas.draw_idle()
            return

        if key == 's':
            filename = f'custom_pattern_{args.grid_size}x{args.grid_size}.npy'
            env.save_pattern(filename)
            print(f"Current grid state saved to {filename}")
            print(f"Density: {np.mean(env.ca_grid):.3f}, Live cells: {np.sum(env.ca_grid)}")
            return

        if key == 'c':
            env.reset()
            step_counter[0] = 0
            action_history.clear()
            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            title_text.set_text(f"Step: 0/{args.steps} | Manual Mode")
            
            # Reset bars
            for bar in bars:
                bar.set_height(0)
            
            metrics_text.set_text("Grid cleared. Ready for new sequence.")
            fig.canvas.draw_idle()
            return

        if key in key_map:
            action = key_map[key]
        elif key in hex_keys:
            action = 5 + hex_keys.index(key)
        else:
            return

        # Execute action
        next_state, done = env.step(action)
        action_history.append(action)
        step_counter[0] += 1

        # Update visualization
        grid_img.set_data(env.ca_grid)
        agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))

        # Update action history bars
        action_counts = np.bincount(list(action_history), minlength=len(action_labels))
        max_count = max(action_counts) if max(action_counts) > 0 else 1
        normalized_counts = action_counts / max_count

        for bar, count in zip(bars, normalized_counts):
            bar.set_height(count)

        # Calculate metrics
        alive_count = int(np.sum(env.ca_grid))
        density = float(np.mean(env.ca_grid))
        
        metrics_str = (
            f"Step: {step_counter[0]}/{args.steps}\n"
            f"Action: {env.actions[action]}\n"
            f"Alive cells: {alive_count}\n"
            f"Density: {density:.3f}\n"
        )
        
        title_str = f"Step: {step_counter[0]}/{args.steps} | Manual Mode"

        if env.target_pattern is not None:
            fitness = env.calculate_fitness(env.ca_grid)
            match_pct = np.mean(env.ca_grid == env.target_pattern)
            metrics_str += (
                f"Fitness: {fitness:.2f}\n"
                f"Match: {match_pct:.1%}\n"
            )
            title_str = f"Step: {step_counter[0]}/{args.steps} | Fitness: {fitness:.2f} | Match: {match_pct:.1%}"
            if step_counter[0] == args.steps:
                metrics_str += f"\nFinal Score: {fitness:.2f}\n"
        
        if step_counter[0] >= args.steps:
            metrics_str += "\nMax steps reached!\nPress 'C' to clear."
        else:
            metrics_str += (
                f"\nControls:\n"
                f"Arrow Keys/Space: Move/Wait\n"
                f"0-F: Write patterns\n"
                f"S: Save grid as pattern\n"
                f"C: Clear grid\n"
                f"Q: Quit"
            )

        metrics_text.set_text(metrics_str)
        title_text.set_text(title_str)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\nManual Play Mode Controls:")
    print("- Arrow Keys: Move agent (↑/↓/←/→)")
    print("- Space: Do nothing")
    print("- 0-F: Write 2x2 patterns (hex notation)")
    print("- S: Save current grid state as pattern file")
    print("- C: Clear grid")
    print("- Q: Quit\n")
    
    # Initial text
    metrics_text.set_text(
        f"Step: 0/{args.steps}\n\n"
        f"Controls:\n"
        f"Arrow Keys/Space: Move/Wait\n"
        f"0-F: Write patterns\n"
        f"S: Save grid as pattern\n"
        f"C: Clear grid\n"
        f"Q: Quit"
    )


    plt.show(block=True)


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evolutionary Algorithm for CA Pattern Matching")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select mode')

    # Training
    train_parser = subparsers.add_parser('train', help='Run evolutionary training')
    train_parser.add_argument('--generations', type=int, default=500, help='Number of generations')
    train_parser.add_argument('--steps', type=int, default=10, help='Length of action sequences')
    train_parser.add_argument('--population-size', type=int, default=100, help='Population size')
    train_parser.add_argument('--elite-fraction', type=float, default=0.2, help='Fraction of elite individuals')
    train_parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation probability per action')
    train_parser.add_argument('--grid-size', type=int, default=12, help='Size of CA grid')
    train_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'])
    train_parser.add_argument('--pattern-file', type=str, required=True, help='Target pattern file')
    train_parser.add_argument('--live-plot', type=int, nargs='?', const=1, default=None,
                             help='Live plotting frequency (e.g., 1 for every gen, 10 for every 10th gen). No value means off.')
    train_parser.add_argument('--save-freq', type=int, default=100, help='Checkpoint save frequency')

    # Demo
    demo_parser = subparsers.add_parser('demo', help='Demonstrate a sequence')
    demo_parser.add_argument('--sequence-file', type=str, default='best_sequence.npy', help='Sequence file')
    demo_parser.add_argument('--pattern-file', type=str, help='Target pattern file')
    demo_parser.add_argument('--grid-size', type=int, default=12)
    demo_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'])

    # Manual Play
    manual_parser = subparsers.add_parser('manual', help='Manual play mode with keyboard control')
    manual_parser.add_argument('--grid-size', type=int, default=12)
    manual_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'])
    manual_parser.add_argument('--pattern-file', type=str, default=None, help='Target pattern file (optional)')
    manual_parser.add_argument('--steps', type=int, default=10, help='Maximum steps before reset is required')

    # Pattern Creator
    pattern_parser = subparsers.add_parser('create_pattern', help='Interactive pattern creator')
    pattern_parser.add_argument('--grid-size', type=int, default=12)

    args = parser.parse_args()

    if args.mode == 'train':
        train_evolutionary(args)
    elif args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'manual':
        run_demo_manual(args)
    elif args.mode == 'create_pattern':
        interactive_pattern_creator(args)
