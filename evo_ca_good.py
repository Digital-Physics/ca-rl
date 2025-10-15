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
    
    def __init__(self, env, sequence_length=10, population_size=100, 
                 elite_fraction=0.2, mutation_rate=0.1):
        self.env = env
        self.sequence_length = sequence_length
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
        
        # Statistics
        self.generation = 0
        self.avg_fitness_history = []
        self.max_fitness_history = []
        self.diversity_history = []
        
    def _random_sequence(self):
        """Generate a random action sequence."""
        return np.random.randint(0, self.num_actions, size=self.sequence_length)
    
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
    
    def evaluate_population(self):
        """Evaluate all sequences in the population."""
        for i in range(self.population_size):
            self.fitness_scores[i], _ = self.evaluate_sequence(self.population[i])
            
            # Track best solution
            if self.fitness_scores[i] > self.best_fitness:
                self.best_fitness = self.fitness_scores[i]
                self.best_sequence = self.population[i].copy()
        
        # Update statistics
        self.avg_fitness_history.append(np.mean(self.fitness_scores))
        self.max_fitness_history.append(np.max(self.fitness_scores))
        self.diversity_history.append(self._calculate_diversity())
        
    def _calculate_diversity(self):
        """Calculate population diversity (unique sequences)."""
        unique = len(set(tuple(seq) for seq in self.population))
        return unique / self.population_size
    
    def select_parents(self):
        """Select elite individuals for breeding."""
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        return [self.population[i].copy() for i in elite_indices]
    
    def crossover(self, parent1, parent2):
        """Single-point crossover between two parents."""
        crossover_point = np.random.randint(1, self.sequence_length)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    
    def mutate(self, sequence):
        """Mutate a sequence by randomly changing some actions."""
        mutated = sequence.copy()
        for i in range(self.sequence_length):
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


def train_evolutionary(args):
    """Main evolutionary training loop."""
    print("--- Starting Evolutionary Training ---")
    
    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, 
                reward_type='pattern', max_steps=args.sequence_length)
    
    # Load pattern
    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)
    else:
        print("Warning: No pattern file loaded. Using empty grid as target.")
    
    optimizer = EvolutionaryOptimizer(
        env=env,
        sequence_length=args.sequence_length,
        population_size=args.population_size,
        elite_fraction=args.elite_fraction,
        mutation_rate=args.mutation_rate
    )
    
    # Setup live plotting
    if args.live_plot:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
        
        # Grid displays
        ax_best = fig.add_subplot(gs[0, 0])
        ax_target = fig.add_subplot(gs[0, 1])
        ax_current = fig.add_subplot(gs[0, 2])
        
        # Metrics
        ax_fitness = fig.add_subplot(gs[0, 3])
        ax_diversity = fig.add_subplot(gs[1, 0])
        ax_actions = fig.add_subplot(gs[1, 1:3])
        ax_dist = fig.add_subplot(gs[1, 3])
        ax_fitness_dist = fig.add_subplot(gs[2, 0])
        ax_info = fig.add_subplot(gs[2, 1:])
        
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
        ax_current.set_title('Current Best', fontweight='bold')
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
        
        # Action sequence visualization
        action_labels = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
        action_colors = plt.cm.tab20(np.linspace(0, 1, len(action_labels)))
        
        # Fitness distribution
        ax_fitness_dist.set_title('Fitness Distribution', fontweight='bold')
        ax_fitness_dist.set_xlabel('Fitness')
        ax_fitness_dist.set_ylabel('Count')
        
        # Action distribution
        ax_dist.set_title('Action Usage', fontweight='bold')
        ax_dist.set_ylabel('Frequency')
        
        # Info text
        info_text = ax_info.text(0.05, 0.95, '', va='top', fontsize=11, family='monospace')
        ax_info.set_title('Statistics', fontweight='bold')
        ax_info.axis('off')
        
        plt.pause(0.1)
    
    # Training loop
    print(f"\nTraining for {args.generations} generations...")
    print(f"Population size: {args.population_size}, Sequence length: {args.sequence_length}")
    print(f"Elite fraction: {args.elite_fraction}, Mutation rate: {args.mutation_rate}\n")
    
    for generation in tqdm(range(args.generations), desc="Evolution"):
        # Evaluate population
        optimizer.evaluate_population()
        
        # Update visualization
        if args.live_plot and (generation % args.plot_freq == 0 or generation == args.generations - 1):
            # Evaluate best sequence for display
            best_fitness, best_grid = optimizer.evaluate_sequence(optimizer.best_sequence)
            
            # Update grid displays
            best_img.set_data(best_grid)
            
            # Evaluate a random current member
            random_idx = np.random.randint(0, optimizer.population_size)
            _, current_grid = optimizer.evaluate_sequence(optimizer.population[random_idx])
            current_img.set_data(current_grid)
            
            # Update fitness plot
            gens = range(1, len(optimizer.max_fitness_history) + 1)
            line_max.set_data(gens, optimizer.max_fitness_history)
            line_avg.set_data(gens, optimizer.avg_fitness_history)
            ax_fitness.relim()
            ax_fitness.autoscale_view()
            
            # Update diversity plot
            line_div.set_data(gens, optimizer.diversity_history)
            
            # Action sequence visualization
            ax_actions.clear()
            colors = [action_colors[a] for a in optimizer.best_sequence]
            ax_actions.bar(range(args.sequence_length), optimizer.best_sequence, color=colors)
            ax_actions.set_title('Best Action Sequence', fontweight='bold')
            ax_actions.set_xlabel('Step')
            ax_actions.set_ylabel('Action ID')
            ax_actions.set_ylim([-1, optimizer.num_actions])
            ax_actions.grid(axis='y', alpha=0.3)
            
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
            
            # Update info text
            info_str = (
                f"Generation: {generation + 1}/{args.generations}\n"
                f"Best Fitness: {optimizer.best_fitness:.2f}\n"
                f"Avg Fitness: {optimizer.avg_fitness_history[-1]:.2f}\n"
                f"Diversity: {optimizer.diversity_history[-1]:.2%}\n"
                f"Perfect Matches: {np.sum(optimizer.fitness_scores >= 100)}\n\n"
                f"Best Sequence:\n"
            )
            
            # Show best action sequence with labels
            for i, action in enumerate(optimizer.best_sequence):
                if i % 5 == 0 and i > 0:
                    info_str += "\n"
                info_str += f"{action_labels[action]:>3} "
            
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
    
    if args.live_plot:
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
    
    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)
    
    # Setup visualization
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax_grid = fig.add_subplot(gs[:, 0])
    ax_target = fig.add_subplot(gs[:, 1])
    ax_actions = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[1, 2])
    
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
    title_text = ax_grid.set_title("Step 0")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    
    # Action sequence
    action_labels = ['↑', '↓', '←', '→', '○'] + [f'{i:X}' for i in range(16)]
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
    
    def update(frame):
        if frame < len(sequence):
            action = sequence[frame]
            state, _ = env.step(action)
            
            grid_img.set_data(state)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            title_text.set_text(f"Step {frame + 1}")
            
            # Highlight current action
            for i, bar in enumerate(bars):
                if i == frame:
                    bar.set_color('red')
                elif i < frame:
                    bar.set_alpha(0.3)
            
            fitness = env.calculate_fitness(state)
            match_pct = np.mean(state == env.target_pattern) if env.target_pattern is not None else 0
            
            info_str = (
                f"Step: {frame + 1}/{len(sequence)}\n"
                f"Action: {action_labels[action]}\n"
                f"Fitness: {fitness:.2f}\n"
                f"Match: {match_pct:.1%}\n"
                f"Alive cells: {np.sum(state)}"
            )
            info_text.set_text(info_str)
        
        return [grid_img, agent_patch]
    
    ani = animation.FuncAnimation(fig, update, frames=len(sequence) + 10, 
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


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evolutionary Algorithm for CA Pattern Matching")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select mode')

    # Training
    train_parser = subparsers.add_parser('train', help='Run evolutionary training')
    train_parser.add_argument('--generations', type=int, default=500, help='Number of generations')
    train_parser.add_argument('--sequence-length', type=int, default=10, help='Length of action sequences')
    train_parser.add_argument('--population-size', type=int, default=100, help='Population size')
    train_parser.add_argument('--elite-fraction', type=float, default=0.2, help='Fraction of elite individuals')
    train_parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation probability per action')
    train_parser.add_argument('--grid-size', type=int, default=12, help='Size of CA grid')
    train_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'])
    train_parser.add_argument('--pattern-file', type=str, required=True, help='Target pattern file')
    train_parser.add_argument('--live-plot', action='store_true', help='Enable live plotting')
    train_parser.add_argument('--plot-freq', type=int, default=10, help='Plot update frequency')
    train_parser.add_argument('--save-freq', type=int, default=100, help='Checkpoint save frequency')

    # Demo
    demo_parser = subparsers.add_parser('demo', help='Demonstrate a sequence')
    demo_parser.add_argument('--sequence-file', type=str, default='best_sequence.npy', help='Sequence file')
    demo_parser.add_argument('--pattern-file', type=str, help='Target pattern file')
    demo_parser.add_argument('--grid-size', type=int, default=12)
    demo_parser.add_argument('--rules', type=str, default='conway', choices=['conway', 'seeds', 'maze'])

    # Pattern Creator
    pattern_parser = subparsers.add_parser('create_pattern', help='Interactive pattern creator')
    pattern_parser.add_argument('--grid-size', type=int, default=12)

    args = parser.parse_args()

    if args.mode == 'train':
        train_evolutionary(args)
    elif args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'create_pattern':
        interactive_pattern_creator(args)
        