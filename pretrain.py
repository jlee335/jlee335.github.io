#!/usr/bin/env python3
"""
Fish Evolution Pre-training System
----------------------------------
Discrete-event simulation to evolve fish neural networks before deployment to website.

Features:
- SimPy-based discrete-event simulation
- 100+ fish for better genetic diversity
- Multiple food sources for richer training
- Antenna-based communication: fish learn to broadcast 3-value signals
- Signal threshold: only signals with magnitude > threshold are transmitted
- Social reward system: fish within comm range get 0.5 points when nearby fish eat food
- Parallel training mode: run K threads with variations in aspect ratios and food counts
- Saves best-performing brains as JSON weights for web deployment
- Optional visualization with matplotlib
"""

import numpy as np
import json
import time
import sys
import simpy
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ─── Configuration ──────────────────────────────────────────────────────
def load_config_from_file(config_path: str = "assets/config.json") -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using defaults")
        return {}

@dataclass
class Config:
    # Simulation
    width: int = 1920
    height: int = 1080
    fish_count: int = 100  # More fish = better evolution (overrides config file for training)
    food_count: int = 3    # Multiple food sources (overrides config file for training)
    generation_time: float = 60.0  # Evolve every 60 seconds (vs 30s on web)
    
    # Neural Network (loaded from config.json)
    input_size: int = 11  # 3 food + 4 social + 1 memory + 3 antenna_in
    hidden_size: int = 8
    output_size: int = 6  # 3 actions + 3 antenna_out
    
    # Physics (loaded from config.json)
    fish_speed: float = 30.0
    damping: float = 0.992
    max_speed: float = 180.0
    tick_dt: float = 0.120  # 120ms brain tick
    physics_dt: float = 0.016  # ~60 FPS physics
    
    # Fish (loaded from config.json)
    fish_radius: float = 6.0
    sense_dist_factor: float = 0.1  # Visual range: 1/10 of max dimension for food spotting
    comm_range_factor: float = 0.15  # Communication range for fish-to-fish
    turn_rad: float = np.pi / 6  # 30 degrees
    signal_threshold: float = 0.5  # Minimum antenna signal magnitude to broadcast
    
    # Food (loaded from config.json)
    food_radius: float = 8.0
    food_speed: float = 45.0
    eat_dist: float = 14.0  # fish_radius + food_radius
    
    # Evolution (loaded from config.json)
    mutation_rate: float = 0.15
    mutation_strength: float = 0.5
    survival_rate: float = 0.5  # Top 50% survive
    
    @classmethod
    def from_file(cls, config_path: str = "assets/config.json", **overrides):
        """Create Config from JSON file with optional overrides"""
        file_config = load_config_from_file(config_path)
        
        # Extract values from loaded config
        kwargs = {}
        
        if file_config:
            # Neural network
            if 'neural_network' in file_config:
                nn = file_config['neural_network']
                kwargs['input_size'] = nn.get('input_size', 5)
                kwargs['hidden_size'] = nn.get('hidden_size', 5)
                kwargs['output_size'] = nn.get('output_size', 3)
            
            # Physics
            if 'physics' in file_config:
                phys = file_config['physics']
                kwargs['fish_speed'] = phys.get('fish_speed', 30.0)
                kwargs['damping'] = phys.get('damping', 0.992)
                kwargs['max_speed'] = phys.get('max_speed', 180.0)
                kwargs['tick_dt'] = phys.get('tick_ms', 120) / 1000.0
                kwargs['physics_dt'] = phys.get('physics_dt', 0.016)
            
            # Fish
            if 'fish' in file_config:
                fish = file_config['fish']
                kwargs['fish_radius'] = fish.get('fish_radius', 6.0)
                kwargs['sense_dist_factor'] = fish.get('visual_range_factor', 0.1)
                kwargs['comm_range_factor'] = fish.get('comm_range_factor', 0.15)
                turn_deg = fish.get('turn_deg', 30)
                kwargs['turn_rad'] = turn_deg * np.pi / 180
            
            # Communication
            if 'communication' in file_config:
                comm = file_config['communication']
                kwargs['signal_threshold'] = comm.get('signal_threshold', 0.5)
            
            # Food
            if 'food' in file_config:
                food = file_config['food']
                kwargs['food_radius'] = food.get('food_radius', 8.0)
                kwargs['food_speed'] = food.get('food_speed', 45.0)
            
            # Evolution
            if 'evolution' in file_config:
                evo = file_config['evolution']
                kwargs['mutation_rate'] = evo.get('mutation_rate', 0.15)
                kwargs['mutation_strength'] = evo.get('mutation_strength', 0.5)
                kwargs['survival_rate'] = evo.get('survival_rate', 0.5)
            
            # Display (for default dimensions)
            if 'display' in file_config:
                disp = file_config['display']
                kwargs['width'] = disp.get('default_width', 1920)
                kwargs['height'] = disp.get('default_height', 1080)
            
            # Simulation (but allow training overrides)
            if 'simulation' in file_config:
                sim = file_config['simulation']
                # Don't override fish_count and food_count by default for training
                kwargs['generation_time'] = sim.get('generation_time', 60.0)
        
        # Calculate derived values
        if 'fish_radius' in kwargs and 'food_radius' in kwargs:
            kwargs['eat_dist'] = kwargs['fish_radius'] + kwargs['food_radius']
        
        # Apply command-line overrides
        kwargs.update(overrides)
        
        return cls(**kwargs)


# ─── Neural Network ─────────────────────────────────────────────────────
class NeuralNet:
    """2-layer feedforward network with tanh activation"""
    
    def __init__(self, config: Config, W1=None, W2=None):
        self.config = config
        if W1 is None or W2 is None:
            # Xavier initialization
            self.W1 = np.random.randn(config.hidden_size, config.input_size + 1) * np.sqrt(2.0 / config.input_size)
            self.W2 = np.random.randn(config.output_size, config.hidden_size + 1) * np.sqrt(2.0 / config.hidden_size)
        else:
            self.W1 = W1
            self.W2 = W2
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        # Add bias
        x = np.append(inputs, 1.0)
        
        # Hidden layer with tanh (better for locomotion than ReLU)
        hidden = np.tanh(self.W1 @ x)
        
        # Add bias to hidden
        hidden = np.append(hidden, 1.0)
        
        # Output layer
        output = self.W2 @ hidden
        return output
    
    def act(self, inputs: np.ndarray) -> tuple[int, np.ndarray]:
        """Get action and antenna signal
        Returns: (action_idx, antenna_signal[3])
        """
        output = self.forward(inputs)
        # First 3 outputs are actions (0=forward, 1=left+forward, 2=right+forward)
        action = int(np.argmax(output[:3]))
        # Last 3 outputs are antenna signal (use tanh to bound between -1 and 1)
        antenna = np.tanh(output[3:6])
        return action, antenna
    
    def mutate(self) -> 'NeuralNet':
        """Create mutated copy"""
        W1_new = self.W1.copy()
        W2_new = self.W2.copy()
        
        # Mutate W1
        mask1 = np.random.random(W1_new.shape) < self.config.mutation_rate
        W1_new[mask1] += np.random.randn(np.sum(mask1)) * self.config.mutation_strength
        
        # Mutate W2
        mask2 = np.random.random(W2_new.shape) < self.config.mutation_rate
        W2_new[mask2] += np.random.randn(np.sum(mask2)) * self.config.mutation_strength
        
        return NeuralNet(self.config, W1_new, W2_new)
    
    def to_dict(self) -> dict:
        """Export weights as dict for JSON serialization"""
        return {
            'W1': self.W1.tolist(),
            'W2': self.W2.tolist()
        }
    
    @classmethod
    def from_dict(cls, config: Config, data: dict) -> 'NeuralNet':
        """Load from dict"""
        return cls(config, np.array(data['W1']), np.array(data['W2']))


# ─── Fish ───────────────────────────────────────────────────────────────
class Fish:
    """A fish agent with neural network controller"""
    
    def __init__(self, config: Config, brain: NeuralNet = None, x=None, y=None):
        self.config = config
        self.x = x if x is not None else np.random.random() * config.width
        self.y = y if y is not None else np.random.random() * config.height
        self.vx = 0.0
        self.vy = 0.0
        self.angle = np.random.random() * 2 * np.pi
        self.brain = brain if brain is not None else NeuralNet(config)
        self.food_eaten = 0
        self.last_action = 0  # Proprioception: memory of previous action
        
        # Antenna communication system
        self.antenna_out = np.zeros(3)  # Signal to broadcast
        self.antenna_in = np.zeros(3)   # Latest received signal (consumed after use)
    
    def get_inputs(self, foods: List['Food'], other_fish: List['Fish']) -> np.ndarray:
        """Get neural network inputs with social awareness and antenna signals"""
        max_dim = max(self.config.width, self.config.height)
        visual_range = max_dim * self.config.sense_dist_factor
        
        # --- 1. Food Inputs ---
        closest_food = None
        min_dist = float('inf')
        
        for food in foods:
            dx = food.x - self.x
            dy = food.y - self.y
            d2 = dx*dx + dy*dy
            if d2 <= visual_range**2 and d2 < min_dist:
                min_dist = d2
                closest_food = food
        
        # Calculate food vector
        if closest_food:
            fx, fy = closest_food.x - self.x, closest_food.y - self.y
            dist_norm = np.sqrt(min_dist) / max_dim
            angle_to_food = np.arctan2(fy, fx) - self.angle
            food_inputs = [np.cos(angle_to_food), np.sin(angle_to_food), dist_norm]
        else:
            food_inputs = [0.0, 0.0, 1.0]  # 1.0 dist means "far"
        
        # --- 2. Neighbor Inputs (Social positioning) ---
        closest_neighbor = None
        min_n_dist = float('inf')
        comm_range_sq = (max_dim * self.config.comm_range_factor) ** 2
        
        for neighbor in other_fish:
            if neighbor is self: continue
            
            dx = neighbor.x - self.x
            dy = neighbor.y - self.y
            d2 = dx*dx + dy*dy
            
            if d2 < comm_range_sq and d2 < min_n_dist:
                min_n_dist = d2
                closest_neighbor = neighbor
        
        if closest_neighbor:
            # Relative Angle to neighbor (Where are they?)
            nx, ny = closest_neighbor.x - self.x, closest_neighbor.y - self.y
            angle_to_neighbor = np.arctan2(ny, nx) - self.angle
            
            # Relative Heading of neighbor (Where are they looking?)
            heading_diff = closest_neighbor.angle - self.angle
            
            neighbor_dist_norm = np.sqrt(min_n_dist) / max_dim
            social_inputs = [
                np.cos(angle_to_neighbor), 
                np.sin(angle_to_neighbor), 
                np.cos(heading_diff),  # Alignment
                neighbor_dist_norm
            ]
        else:
            social_inputs = [0.0, 0.0, 0.0, 1.0]
        
        # --- 3. Proprioception (Memory) ---
        memory_input = [self.last_action - 1]  # -1, 0, or 1
        
        # --- 4. Antenna Input (Received signal) ---
        antenna_inputs = self.antenna_in.tolist()
        
        # Combine: 3 (Food) + 4 (Social) + 1 (Memory) + 3 (Antenna) = 11 inputs
        return np.array(food_inputs + social_inputs + memory_input + antenna_inputs)
    
    def push(self, foods: List['Food'], other_fish: List['Fish']):
        """Apply brain decision as impulse and broadcast antenna signal"""
        inputs = self.get_inputs(foods, other_fish)
        action, antenna_signal = self.brain.act(inputs)
        
        # Turn
        if action == 1:
            self.angle -= self.config.turn_rad
        elif action == 2:
            self.angle += self.config.turn_rad
        
        # Apply forward impulse
        self.vx += np.cos(self.angle) * self.config.fish_speed
        self.vy += np.sin(self.angle) * self.config.fish_speed
        
        # Store action for next tick (proprioception)
        self.last_action = action
        
        # Broadcast antenna signal (fish learns to output strong signals to communicate)
        self.antenna_out = antenna_signal
        
        # Clear received signal after consumption
        self.antenna_in = np.zeros(3)
    
    def get_signal_magnitude(self) -> float:
        """Calculate L2 norm of antenna output"""
        return np.linalg.norm(self.antenna_out)
    
    def physics(self, dt: float, foods: List['Food']) -> bool:
        """Update position and handle collisions
        
        Returns:
            bool: True if this fish ate food this frame
        """
        ate_food = False
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Damping
        self.vx *= self.config.damping
        self.vy *= self.config.damping
        
        # Speed limit
        speed = np.sqrt(self.vx*self.vx + self.vy*self.vy)
        if speed > self.config.max_speed:
            self.vx = (self.vx / speed) * self.config.max_speed
            self.vy = (self.vy / speed) * self.config.max_speed
        
        # Wall bounce
        if self.x < 0:
            self.x = 0
            self.vx = -self.vx
            self.angle = np.arctan2(self.vy, self.vx)
        elif self.x > self.config.width:
            self.x = self.config.width
            self.vx = -self.vx
            self.angle = np.arctan2(self.vy, self.vx)
        
        if self.y < 0:
            self.y = 0
            self.vy = -self.vy
            self.angle = np.arctan2(self.vy, self.vx)
        elif self.y > self.config.height:
            self.y = self.config.height
            self.vy = -self.vy
            self.angle = np.arctan2(self.vy, self.vx)
        
        # Check food collisions (bounce instead of respawn)
        for food in foods:
            dx = food.x - self.x
            dy = food.y - self.y
            if dx*dx + dy*dy < self.config.eat_dist * self.config.eat_dist:
                dist = np.sqrt(dx*dx + dy*dy) or 1e-6
                nx = dx / dist
                ny = dy / dist

                # Push apart to avoid overlap
                overlap = self.config.eat_dist - dist
                self.x -= nx * overlap * 0.5
                self.y -= ny * overlap * 0.5
                food.x += nx * overlap * 0.5
                food.y += ny * overlap * 0.5

                # Elastic collision (equal mass)
                restitution = 0.9
                rvx = food.vx - self.vx
                rvy = food.vy - self.vy
                vel_along_normal = rvx * nx + rvy * ny
                if vel_along_normal < 0:
                    impulse = -(1 + restitution) * vel_along_normal / 2
                    self.vx -= impulse * nx
                    self.vy -= impulse * ny
                    food.vx += impulse * nx
                    food.vy += impulse * ny

                # Clamp back into bounds after collision resolution
                self.x = min(max(self.x, 0), self.config.width)
                self.y = min(max(self.y, 0), self.config.height)
                food.x = min(max(food.x, 0), self.config.width)
                food.y = min(max(food.y, 0), self.config.height)

                if not food.eaten_this_frame:
                    self.food_eaten += 1
                    food.eaten_this_frame = True
                    ate_food = True
        
        return ate_food


# ─── Food ───────────────────────────────────────────────────────────────
class Food:
    """Autonomous food pellet that bounces around"""
    
    def __init__(self, config: Config):
        self.config = config
        self.eaten_this_frame = False
        self.respawn(config)
    
    def respawn(self, config: Config):
        """Respawn at random location with random velocity"""
        margin = 40
        self.x = margin + np.random.random() * (config.width - 2*margin)
        self.y = margin + np.random.random() * (config.height - 2*margin)
        angle = np.random.random() * 2 * np.pi
        self.vx = np.cos(angle) * config.food_speed
        self.vy = np.sin(angle) * config.food_speed
        self.eaten_this_frame = False
    
    def physics(self, dt: float):
        """Update position and bounce off walls"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Wall bounce
        if self.x < 0 or self.x > self.config.width:
            self.vx = -self.vx
            self.x = max(0, min(self.config.width, self.x))
        
        if self.y < 0 or self.y > self.config.height:
            self.vy = -self.vy
            self.y = max(0, min(self.config.height, self.y))


# ─── Simulation ─────────────────────────────────────────────────────────
class Simulation:
    """SimPy-based discrete-event simulation engine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.env = simpy.Environment()
        self.fish: List[Fish] = []
        self.foods: List[Food] = []
        self.generation = 1
        self.time_in_gen = 0.0
        self.generation_end_time = config.generation_time
        
        # Initialize population
        for _ in range(config.fish_count):
            self.fish.append(Fish(config))
        
        # Initialize food
        for _ in range(config.food_count):
            self.foods.append(Food(config))
    
    def brain_tick_process(self):
        """Process for fish brain decisions and antenna communication"""
        while True:
            # Step 1: All fish make decisions and broadcast antenna signals
            for fish in self.fish:
                fish.push(self.foods, self.fish)
            
            # Step 2: Propagate antenna signals to neighbors (latest signal only)
            max_dim = max(self.config.width, self.config.height)
            comm_range_sq = (max_dim * self.config.comm_range_factor) ** 2
            
            for receiver in self.fish:
                latest_signal = None
                
                # Find latest signal from neighbors within comm range
                for broadcaster in self.fish:
                    if broadcaster is receiver:
                        continue
                    
                    dx = broadcaster.x - receiver.x
                    dy = broadcaster.y - receiver.y
                    d2 = dx*dx + dy*dy
                    
                    if d2 < comm_range_sq:
                        # Check if signal magnitude exceeds threshold
                        magnitude = broadcaster.get_signal_magnitude()
                        
                        if magnitude >= self.config.signal_threshold:
                            # Within range AND strong enough signal
                            # Last one wins in iteration order
                            latest_signal = broadcaster.antenna_out
                
                # Set received signal (if any)
                if latest_signal is not None:
                    receiver.antenna_in = latest_signal.copy()
            
            # Wait for next brain tick
            yield self.env.timeout(self.config.tick_dt)
    
    def physics_process(self):
        """Process for physics updates with social reward system"""
        while True:
            dt = self.config.physics_dt
            
            # Mark all food as not eaten this frame
            for food in self.foods:
                food.eaten_this_frame = False
            
            # Update all fish physics and track which fish ate food
            fish_that_ate = []
            for fish in self.fish:
                if fish.physics(dt, self.foods):
                    fish_that_ate.append(fish)
            
            # Social reward: Give 0.5 points to fish within comm range of successful foragers
            if fish_that_ate:
                max_dim = max(self.config.width, self.config.height)
                comm_range_sq = (max_dim * self.config.comm_range_factor) ** 2
                
                for successful_fish in fish_that_ate:
                    # Reward all nearby fish (but not the successful fish itself)
                    for nearby_fish in self.fish:
                        if nearby_fish is successful_fish:
                            continue
                        
                        dx = nearby_fish.x - successful_fish.x
                        dy = nearby_fish.y - successful_fish.y
                        d2 = dx*dx + dy*dy
                        
                        if d2 < comm_range_sq:
                            # Within communication range - give social reward
                            nearby_fish.food_eaten += 0.5
            
            # Update food physics (no respawn on contact)
            for food in self.foods:
                food.physics(dt)
            
            # Wait for next physics step
            yield self.env.timeout(dt)
    
    def run_generation(self):
        """Run one generation using SimPy"""
        # Reset simulation time for this generation
        self.env = simpy.Environment()
        self.time_in_gen = 0.0
        
        # Start processes
        self.env.process(self.brain_tick_process())
        self.env.process(self.physics_process())
        
        # Run until generation time
        self.env.run(until=self.config.generation_time)
        self.time_in_gen = self.env.now
    
    def evolve(self):
        """Evolutionary step with tournament selection and crossover"""
        # Sort by fitness
        self.fish.sort(key=lambda f: f.food_eaten, reverse=True)
        
        # Get statistics
        avg_fitness = np.mean([f.food_eaten for f in self.fish])
        max_fitness = self.fish[0].food_eaten
        
        # Create new generation
        new_fish = []
        population = self.fish
        pop_size = len(population)
        
        # Elitism: Keep absolute best 2 fish unchanged
        new_fish.append(Fish(self.config, population[0].brain))
        new_fish.append(Fish(self.config, population[1].brain))
        
        # Tournament selection + Crossover
        while len(new_fish) < self.config.fish_count:
            # Tournament: Pick 3, take best
            p1 = max(np.random.choice(population, 3, replace=False), key=lambda f: f.food_eaten)
            p2 = max(np.random.choice(population, 3, replace=False), key=lambda f: f.food_eaten)
            
            # Crossover (Mix weights)
            child_brain = self.crossover_brains(p1.brain, p2.brain)
            
            # Mutation
            child_brain = child_brain.mutate()
            
            new_fish.append(Fish(self.config, child_brain))
        
        self.fish = new_fish
        self.generation += 1
        
        # Respawn all food at start of new generation
        for food in self.foods:
            food.respawn(self.config)
        
        return avg_fitness, max_fitness
    
    def crossover_brains(self, b1: NeuralNet, b2: NeuralNet) -> NeuralNet:
        """Uniform Crossover: randomly select weights from parent A or B"""
        # Create mask for W1
        mask1 = np.random.rand(*b1.W1.shape) > 0.5
        new_W1 = np.where(mask1, b1.W1, b2.W1)
        
        # Create mask for W2
        mask2 = np.random.rand(*b1.W2.shape) > 0.5
        new_W2 = np.where(mask2, b1.W2, b2.W2)
        
        return NeuralNet(self.config, new_W1, new_W2)
    
    def get_best_brains(self, n: int = 20) -> List[NeuralNet]:
        """Get top N brains by fitness"""
        self.fish.sort(key=lambda f: f.food_eaten, reverse=True)
        return [fish.brain for fish in self.fish[:n]]


# ─── Visualization ──────────────────────────────────────────────────────
class Visualizer:
    """Real-time visualization using matplotlib"""
    
    def __init__(self, sim: Simulation):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            self.plt = plt
            self.has_viz = True
        except ImportError:
            print("Warning: matplotlib not installed, visualization disabled")
            print("Install with: pip install matplotlib")
            self.has_viz = False
            return
        
        self.sim = sim
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(0, sim.config.width)
        self.ax.set_ylim(0, sim.config.height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#0a0a0f')
        self.fig.patch.set_facecolor('#0a0a0f')
        
        # Add space at top for title
        self.fig.subplots_adjust(top=0.95)
        
        # Initialize plot elements
        self.fish_scatter = self.ax.scatter([], [], c='skyblue', s=30, alpha=0.6)
        self.top_fish_scatter = self.ax.scatter([], [], c='gold', s=50, alpha=0.8, edgecolors='orange', linewidth=1.5)
        self.food_scatter = self.ax.scatter([], [], c='salmon', s=100, alpha=0.7, edgecolors='darkred', linewidth=2)
        
        # Title at top center with more space
        self.title = self.ax.set_title('', color='white', fontsize=16, fontweight='bold', pad=15)
        
        # Stats box at top right to avoid overlap
        self.stats_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes,
                                       horizontalalignment='right', verticalalignment='top',
                                       color='lightgray', fontsize=10, family='monospace',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, pad=0.5))
        
        self.ax.tick_params(colors='gray')
        for spine in self.ax.spines.values():
            spine.set_color('gray')
        
        self.frame_count = 0
        self.last_update = time.time()
    
    def draw(self):
        """Passively draw current simulation state"""
        if not self.has_viz:
            return
        
        # Get fish positions
        fish_x = [f.x for f in self.sim.fish]
        fish_y = [f.y for f in self.sim.fish]
        
        # Get top 5 fish
        sorted_fish = sorted(self.sim.fish, key=lambda f: f.food_eaten, reverse=True)
        top_fish = sorted_fish[:5]
        top_x = [f.x for f in top_fish]
        top_y = [f.y for f in top_fish]
        
        # Food positions
        food_x = [f.x for f in self.sim.foods]
        food_y = [f.y for f in self.sim.foods]
        
        # Update scatter plots
        self.fish_scatter.set_offsets(np.c_[fish_x, fish_y])
        self.top_fish_scatter.set_offsets(np.c_[top_x, top_y])
        self.food_scatter.set_offsets(np.c_[food_x, food_y])
        
        # Update stats
        avg_fitness = np.mean([f.food_eaten for f in self.sim.fish])
        max_fitness = sorted_fish[0].food_eaten if sorted_fish else 0
        
        now = time.time()
        fps = 1.0 / (now - self.last_update) if now > self.last_update else 0
        self.last_update = now
        
        # Update title
        self.ax.set_title(f'🐟 Generation {self.sim.generation} 🐟', color='white', fontsize=16, fontweight='bold', pad=15)
        
        stats = f'Avg Fitness: {avg_fitness:.1f}\n'
        stats += f'Max Fitness: {max_fitness}\n'
        stats += f'Time in Gen: {self.sim.time_in_gen:.1f}s / {self.sim.config.generation_time:.1f}s\n'
        stats += f'FPS: {fps:.1f}'
        
        self.stats_text.set_text(stats)
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def show(self):
        """Setup interactive mode for visualization"""
        if not self.has_viz:
            return
        
        self.plt.ion()  # Turn on interactive mode
        self.plt.show()


# ─── Training ───────────────────────────────────────────────────────────
def train(generations: int = 100, config: Config = None, 
          verbose: bool = True, visualize: bool = False):
    """Run discrete-event simulation training"""
    if config is None:
        config = Config.from_file()
    
    sim = Simulation(config)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🐟 FISH NEURAL NETWORK PRE-TRAINING 🐟".center(70))
        print(f"{'='*70}")
        print(f"  Config loaded:   assets/config.json")
        print(f"  Fish:            {config.fish_count}")
        print(f"  Food sources:    {config.food_count}")
        print(f"  Gen time:        {config.generation_time:.3f}s")
        print(f"  Brain tick:      {config.tick_dt*1000:.2f}ms")
        print(f"  Physics step:    {config.physics_dt*1000:.2f}ms")
        print(f"  Generations:     {generations}")
        print(f"  Visualization:   {'ON' if visualize else 'OFF'}")
        print(f"{'='*70}\n")
    
    # Setup visualization if requested
    viz = None
    if visualize:
        viz = Visualizer(sim)
        if viz.has_viz:
            if verbose:
                print("Starting visualization...\n")
            viz.show()
        else:
            if verbose:
                print("Matplotlib not available, falling back to headless training...\n")
            viz = None
    
    # Training loop
    start_time = time.time()
    
    for gen in range(generations):
        gen_start = time.time()
        
        # Run one generation (SimPy handles the discrete events)
        sim.run_generation()
        
        # Update visualization periodically during generation
        if viz:
            viz.draw()
        
        gen_time = time.time() - gen_start
        avg_fitness, max_fitness = sim.evolve()
        
        # Update visualization after evolution
        if viz:
            viz.draw()
        
        # Print generation summary (only if verbose)
        if verbose:
            # Calculate progress stats
            elapsed = time.time() - start_time
            rate = (gen + 1) / elapsed if elapsed > 0 else 0
            eta = (generations - gen - 1) / rate if rate > 0 else 0
            
            overall_progress = (gen + 1) / generations
            overall_bar_length = 30
            overall_filled = int(overall_bar_length * overall_progress)
            overall_bar = '█' * overall_filled + '░' * (overall_bar_length - overall_filled)
            
            print(f"Gen {sim.generation:4d} [{overall_bar}] {overall_progress*100:5.1f}% | "
                  f"Avg: {avg_fitness:6.2f} | Max: {max_fitness:3.0f} | "
                  f"GenTime: {gen_time:.3f}s | Rate: {rate:5.1f} g/s | ETA: {eta:5.1f}s")
            sys.stdout.flush()
    
    elapsed = time.time() - start_time
    total_sim_time = sim.generation * config.generation_time
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ TRAINING COMPLETE".center(70))
        print(f"{'='*70}")
        print(f"  Generations:     {sim.generation}")
        print(f"  Real time:       {elapsed:.1f}s")
        print(f"  Sim time:        {total_sim_time:.1f}s")
        print(f"  Rate:            {sim.generation/elapsed:.1f} gen/s")
        print(f"  Best fitness:    {max([f.food_eaten for f in sim.fish])}")
        print(f"{'='*70}\n")
    
    # Keep visualization window open if enabled
    if viz:
        if verbose:
            print("Visualization window open. Close window to exit.")
        viz.plt.ioff()  # Turn off interactive mode
        viz.plt.show()  # Block until window is closed
    
    return sim


def save_weights(sim: Simulation, filename: str = "pretrained_weights.json", n_brains: int = 20):
    """Save best brains to JSON file"""
    brains = sim.get_best_brains(n_brains)
    
    data = {
        'generation': sim.generation,
        'config': {
            'input_size': sim.config.input_size,
            'hidden_size': sim.config.hidden_size,
            'output_size': sim.config.output_size,
        },
        'brains': [brain.to_dict() for brain in brains]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    file_size = len(json.dumps(data)) / 1024  # KB
    
    print(f"💾 Saved weights to {filename}")
    print(f"  Brains:      {n_brains}")
    print(f"  Generation:  {sim.generation}")
    print(f"  File size:   {file_size:.1f} KB")
    print(f"\n✓ Ready to deploy to website! Copy to assets/ folder.\n")


# ─── Parallel Training ─────────────────────────────────────────────────
def generate_config_variants(base_config: Config, aspect_ratios: List[Tuple[int, int]], 
                            food_counts: List[int]) -> List[Tuple[str, Config]]:
    """Generate configuration variants with different aspect ratios and food counts
    
    Args:
        base_config: Base configuration to vary
        aspect_ratios: List of (width_ratio, height_ratio) tuples, e.g. [(16,9), (4,3)]
        food_counts: List of food counts to try, e.g. [1, 2, 3]
    
    Returns:
        List of (variant_name, config) tuples
    """
    variants = []
    
    # Use a base dimension (e.g., height = 1080) and vary width based on aspect ratio
    base_height = 1080
    
    for width_ratio, height_ratio in aspect_ratios:
        # Calculate width to maintain aspect ratio
        width = int(base_height * width_ratio / height_ratio)
        
        for food_count in food_counts:
            # Create variant name
            variant_name = f"ar{width_ratio}x{height_ratio}_food{food_count}"
            
            # Create config variant
            variant_config = Config.from_file(
                width=width,
                height=base_height,
                food_count=food_count,
                fish_count=base_config.fish_count,
                generation_time=base_config.generation_time
            )
            
            variants.append((variant_name, variant_config))
    
    return variants


def run_single_generation_worker(config_dict: dict, fish_brains: List[dict]) -> Tuple[List[dict], float, float]:
    """Worker function to run one generation (creates sim from scratch to avoid pickling issues)
    
    Args:
        config_dict: Configuration as dictionary
        fish_brains: List of brain dictionaries to load
    
    Returns:
        Tuple of (fish_brain_dicts, avg_fitness, max_fitness)
    """
    # Reconstruct config from dict
    config = Config(
        width=config_dict['width'],
        height=config_dict['height'],
        food_count=config_dict['food_count'],
        fish_count=config_dict['fish_count'],
        generation_time=config_dict['generation_time'],
        input_size=config_dict['input_size'],
        hidden_size=config_dict['hidden_size'],
        output_size=config_dict['output_size'],
        fish_speed=config_dict['fish_speed'],
        damping=config_dict['damping'],
        max_speed=config_dict['max_speed'],
        tick_dt=config_dict['tick_dt'],
        physics_dt=config_dict['physics_dt'],
        fish_radius=config_dict['fish_radius'],
        sense_dist_factor=config_dict['sense_dist_factor'],
        comm_range_factor=config_dict['comm_range_factor'],
        turn_rad=config_dict['turn_rad'],
        signal_threshold=config_dict['signal_threshold'],
        food_radius=config_dict['food_radius'],
        food_speed=config_dict['food_speed'],
        eat_dist=config_dict['eat_dist'],
        mutation_rate=config_dict['mutation_rate'],
        mutation_strength=config_dict['mutation_strength'],
        survival_rate=config_dict['survival_rate']
    )
    
    # Create simulation
    sim = Simulation(config)
    
    # Load fish brains
    sim.fish.clear()
    for brain_dict in fish_brains:
        brain = NeuralNet.from_dict(config, brain_dict)
        fish = Fish(config, brain)
        sim.fish.append(fish)
    
    # Run generation
    sim.run_generation()
    
    # Get fitness stats
    sim.fish.sort(key=lambda f: f.food_eaten, reverse=True)
    avg_fitness = np.mean([f.food_eaten for f in sim.fish])
    max_fitness = sim.fish[0].food_eaten
    
    # Return fish brains as dicts along with their fitness
    fish_data = []
    for fish in sim.fish:
        fish_data.append({
            'brain': fish.brain.to_dict(),
            'fitness': fish.food_eaten
        })
    
    return fish_data, avg_fitness, max_fitness


def config_to_dict(config: Config) -> dict:
    """Convert Config object to dictionary for serialization"""
    return {
        'width': config.width,
        'height': config.height,
        'food_count': config.food_count,
        'fish_count': config.fish_count,
        'generation_time': config.generation_time,
        'input_size': config.input_size,
        'hidden_size': config.hidden_size,
        'output_size': config.output_size,
        'fish_speed': config.fish_speed,
        'damping': config.damping,
        'max_speed': config.max_speed,
        'tick_dt': config.tick_dt,
        'physics_dt': config.physics_dt,
        'fish_radius': config.fish_radius,
        'sense_dist_factor': config.sense_dist_factor,
        'comm_range_factor': config.comm_range_factor,
        'turn_rad': config.turn_rad,
        'signal_threshold': config.signal_threshold,
        'food_radius': config.food_radius,
        'food_speed': config.food_speed,
        'eat_dist': config.eat_dist,
        'mutation_rate': config.mutation_rate,
        'mutation_strength': config.mutation_strength,
        'survival_rate': config.survival_rate
    }


def distribute_brains_to_variants(n_variants: int, global_fish_data: List[dict], fish_per_variant: int) -> List[List[dict]]:
    """Distribute global gene pool evenly across all variants
    
    Args:
        n_variants: Number of variants
        global_fish_data: Sorted list of fish data (best first), each with 'brain' and 'fitness'
        fish_per_variant: Number of fish per variant
    
    Returns:
        List of brain lists, one for each variant
    """
    variant_brains = [[] for _ in range(n_variants)]
    
    # Distribute fish round-robin to ensure each variant gets a mix of fitness levels
    for i in range(fish_per_variant):
        for v in range(n_variants):
            fish_idx = i * n_variants + v
            if fish_idx < len(global_fish_data):
                variant_brains[v].append(global_fish_data[fish_idx]['brain'])
            else:
                # Wrap around to top performers if we run out
                wrap_idx = fish_idx % min(10, len(global_fish_data))
                variant_brains[v].append(global_fish_data[wrap_idx]['brain'])
    
    return variant_brains


def train_parallel(generations: int = 100, base_config: Config = None,
                  aspect_ratios: List[Tuple[int, int]] = None,
                  food_counts: List[int] = None,
                  n_threads: int = None,
                  output_dir: str = "pretrained_variants") -> List[dict]:
    """Run parallel training with shared global gene pool across variants
    
    All variants run in lock-step. After each generation:
    1. All fish from all variants are pooled together
    2. Fish are ranked globally by fitness
    3. Best fish are evolved and redistributed to all variants
    4. This creates a shared gene pool where the best genes propagate across all configurations
    
    Args:
        generations: Number of generations to train
        base_config: Base configuration to vary
        aspect_ratios: List of aspect ratios to try
        food_counts: List of food counts to try
        n_threads: Number of parallel threads (defaults to CPU count)
        output_dir: Directory to save variant weights
    
    Returns:
        List of result dictionaries
    """
    if base_config is None:
        base_config = Config.from_file()
    
    if aspect_ratios is None:
        # Default aspect ratio variations
        aspect_ratios = [
            (16, 9),   # Standard widescreen
            (4, 3),    # Classic
            (21, 9),   # Ultra-wide
            (1, 1),    # Square
        ]
    
    if food_counts is None:
        food_counts = [1, 2, 3]
    
    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    
    # Generate all variants
    variants = generate_config_variants(base_config, aspect_ratios, food_counts)
    
    print(f"\n{'='*80}")
    print(f"🐟 PARALLEL FISH EVOLUTION WITH GLOBAL GENE POOL 🐟".center(80))
    print(f"{'='*80}")
    print(f"  Variants:        {len(variants)} configurations")
    print(f"  Aspect ratios:   {', '.join([f'{w}:{h}' for w, h in aspect_ratios])}")
    print(f"  Food counts:     {', '.join(map(str, food_counts))}")
    print(f"  Threads:         {n_threads} parallel workers")
    print(f"  Generations:     {generations}")
    print(f"  Output dir:      {output_dir}/")
    print(f"  Gene pool:       SHARED across all variants")
    print(f"{'='*80}\n")
    
    # Print all variants
    print("Variants to train:")
    for i, (name, cfg) in enumerate(variants, 1):
        print(f"  {i:2d}. {name:20s} - {cfg.width:4d}x{cfg.height:4d}, {cfg.food_count} food")
    print()
    
    # Convert configs to dictionaries for serialization
    variant_names = []
    variant_configs = []
    for name, cfg in variants:
        variant_names.append(name)
        variant_configs.append(config_to_dict(cfg))
    
    # Initialize random population for each variant
    fish_per_variant = base_config.fish_count
    variant_fish_brains = []
    
    for cfg in variant_configs:
        # Create temporary config to initialize brains
        temp_config = Config(
            input_size=cfg['input_size'],
            hidden_size=cfg['hidden_size'],
            output_size=cfg['output_size'],
            mutation_rate=cfg['mutation_rate'],
            mutation_strength=cfg['mutation_strength']
        )
        
        # Initialize random brains
        brains = []
        for _ in range(fish_per_variant):
            brain = NeuralNet(temp_config)
            brains.append(brain.to_dict())
        variant_fish_brains.append(brains)
    
    # Training loop with shared gene pool
    start_time = time.time()
    generation_stats = []
    current_generation = 1
    
    for gen in range(generations):
        gen_start = time.time()
        
        # Run one generation for all variants in parallel
        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for i in range(len(variants)):
                future = executor.submit(
                    run_single_generation_worker,
                    variant_configs[i],
                    variant_fish_brains[i]
                )
                futures.append(future)
            
            # Collect results
            variant_results = [future.result() for future in futures]
        
        # Collect fitness stats per variant and pool all fish
        variant_stats = []
        global_fish_data = []
        
        for i, (fish_data, avg_fit, max_fit) in enumerate(variant_results):
            variant_stats.append({
                'variant': variant_names[i],
                'avg_fitness': avg_fit,
                'max_fitness': max_fit
            })
            # Add all fish from this variant to global pool
            global_fish_data.extend(fish_data)
        
        # Sort by fitness globally
        global_fish_data.sort(key=lambda f: f['fitness'], reverse=True)
        
        # Get global stats
        global_avg = np.mean([f['fitness'] for f in global_fish_data])
        global_max = global_fish_data[0]['fitness']
        global_top10_avg = np.mean([f['fitness'] for f in global_fish_data[:10]])
        
        # Perform GLOBAL evolution (evolve the combined gene pool)
        population_size = len(global_fish_data)
        
        # Create temporary config for evolution
        base_cfg = variant_configs[0]
        temp_config = Config(
            input_size=base_cfg['input_size'],
            hidden_size=base_cfg['hidden_size'],
            output_size=base_cfg['output_size'],
            mutation_rate=base_cfg['mutation_rate'],
            mutation_strength=base_cfg['mutation_strength']
        )
        
        # Reconstruct fish brains for evolution
        global_fish_brains = [NeuralNet.from_dict(temp_config, f['brain']) for f in global_fish_data]
        
        # Create new global generation
        new_global_fish_data = []
        
        # Elitism: Keep absolute best fish unchanged
        elite_count = min(20, population_size // 20)
        for i in range(elite_count):
            new_global_fish_data.append({
                'brain': global_fish_data[i]['brain'],
                'fitness': 0  # Will be evaluated next generation
            })
        
        # Tournament selection + Crossover for the rest
        while len(new_global_fish_data) < population_size:
            # Tournament: Pick 3 from top 50%, take best
            top_half_size = population_size // 2
            tournament_size = min(3, top_half_size)
            
            p1_idx = max(np.random.choice(top_half_size, tournament_size, replace=False),
                        key=lambda idx: global_fish_data[idx]['fitness'])
            p2_idx = max(np.random.choice(top_half_size, tournament_size, replace=False),
                        key=lambda idx: global_fish_data[idx]['fitness'])
            
            p1_brain = global_fish_brains[p1_idx]
            p2_brain = global_fish_brains[p2_idx]
            
            # Crossover
            mask1 = np.random.rand(*p1_brain.W1.shape) > 0.5
            new_W1 = np.where(mask1, p1_brain.W1, p2_brain.W1)
            
            mask2 = np.random.rand(*p1_brain.W2.shape) > 0.5
            new_W2 = np.where(mask2, p1_brain.W2, p2_brain.W2)
            
            child_brain = NeuralNet(temp_config, new_W1, new_W2)
            
            # Mutation
            child_brain = child_brain.mutate()
            
            new_global_fish_data.append({
                'brain': child_brain.to_dict(),
                'fitness': 0
            })
        
        # Redistribute evolved population to all variants
        variant_fish_brains = distribute_brains_to_variants(
            len(variants),
            new_global_fish_data,
            fish_per_variant
        )
        
        current_generation += 1
        gen_time = time.time() - gen_start
        
        # Store generation statistics
        generation_stats.append({
            'generation': current_generation,
            'global_avg': global_avg,
            'global_max': global_max,
            'global_top10_avg': global_top10_avg,
            'variants': variant_stats,
            'time': gen_time
        })
        
        # Print progress
        elapsed = time.time() - start_time
        rate = (gen + 1) / elapsed if elapsed > 0 else 0
        eta = (generations - gen - 1) / rate if rate > 0 else 0
        
        progress = (gen + 1) / generations
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"Gen {current_generation:4d} [{bar}] {progress*100:5.1f}% | "
              f"Global Avg: {global_avg:6.2f} | Max: {global_max:3.0f} | "
              f"Top10: {global_top10_avg:6.2f} | "
              f"Time: {gen_time:.2f}s | ETA: {eta:.0f}s")
        sys.stdout.flush()
    
    total_time = time.time() - start_time
    
    # Run final generation to evaluate final fitness across all variants
    print("\nEvaluating final global gene pool across all variants...")
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(len(variants)):
            future = executor.submit(
                run_single_generation_worker,
                variant_configs[i],
                variant_fish_brains[i]
            )
            futures.append(future)
        
        final_results = [future.result() for future in futures]
    
    # Collect ALL fish from final evaluation and rank globally
    all_final_fish = []
    variant_summary = []
    
    for i, (name, cfg_dict) in enumerate(zip(variant_names, variant_configs)):
        fish_data, avg_fitness, max_fitness = final_results[i]
        
        # Add to global pool with variant info
        for f in fish_data:
            f['variant'] = name
        all_final_fish.extend(fish_data)
        
        variant_summary.append({
            'variant_name': name,
            'config': {
                'width': cfg_dict['width'],
                'height': cfg_dict['height'],
                'food_count': cfg_dict['food_count'],
            },
            'generations': current_generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness
        })
    
    # Sort globally to find the absolute best brains across all variants
    all_final_fish.sort(key=lambda f: f['fitness'], reverse=True)
    
    # Get top 20 brains globally (best of the best!)
    best_global_brains = [f['brain'] for f in all_final_fish[:20]]
    
    # Print which variants contributed to the top brains
    print("\nTop 20 brains by variant origin:")
    variant_counts = {}
    for i, f in enumerate(all_final_fish[:20], 1):
        variant = f['variant']
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
        print(f"  #{i:2d}: {f['variant']:20s} (fitness: {f['fitness']:.2f})")
    
    print("\nVariant contribution to top 20:")
    for variant, count in sorted(variant_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {variant:20s}: {count:2d} fish ({count*5:.0f}%)")
    
    # Save the best global brains as pretrained_weights.json
    Path(output_dir).mkdir(exist_ok=True)
    output_file = f"{output_dir}/pretrained_weights.json"
    
    base_cfg = variant_configs[0]
    data = {
        'generation': current_generation,
        'training_mode': 'parallel_global_gene_pool',
        'variants_trained': len(variants),
        'total_fish_evaluated': len(all_final_fish),
        'config': {
            'input_size': base_cfg['input_size'],
            'hidden_size': base_cfg['hidden_size'],
            'output_size': base_cfg['output_size'],
        },
        'brains': best_global_brains
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    file_size = len(json.dumps(data)) / 1024
    
    results = variant_summary
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"✓ PARALLEL TRAINING WITH SHARED GENE POOL COMPLETE".center(80))
    print(f"{'='*80}")
    print(f"  Variants:        {len(variants)}")
    print(f"  Generations:     {generations}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Avg per gen:     {total_time/generations:.2f}s")
    print(f"  Global max fit:  {all_final_fish[0]['fitness']:.2f}")
    print(f"  Global avg fit:  {np.mean([f['fitness'] for f in all_final_fish]):.2f}")
    print(f"{'='*80}\n")
    
    # Print detailed results table
    print("Final variant performance:")
    print(f"{'Variant':<25} {'Dimensions':<15} {'Food':>5} {'Max Fit':>8} {'Avg Fit':>8}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['max_fitness'], reverse=True):
        cfg = r['config']
        dims = f"{cfg['width']}x{cfg['height']}"
        print(f"{r['variant_name']:<25} {dims:<15} {cfg['food_count']:>5} "
              f"{r['max_fitness']:>8.0f} {r['avg_fitness']:>8.2f}")
    
    print(f"\n💾 Best global brains saved!")
    print(f"  File:            {output_file}")
    print(f"  Brains:          20 (best from {len(all_final_fish)} total fish)")
    print(f"  Generation:      {current_generation}")
    print(f"  File size:       {file_size:.1f} KB")
    
    # Save detailed training summary
    summary_file = f"{output_dir}/training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_variants': len(variants),
            'total_time': total_time,
            'generations': generations,
            'shared_gene_pool': True,
            'output_file': output_file,
            'top_20_variant_sources': variant_counts,
            'generation_stats': generation_stats,
            'final_results': results
        }, f, indent=2)
    
    print(f"📊 Training summary:  {summary_file}")
    print(f"\n✓ Ready to deploy! Copy {output_file} to assets/ folder.\n")
    
    return results


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    """Run pre-training and save weights"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pre-train fish neural networks using discrete-event simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (default)
  python pretrain.py
  
  # Parallel training with variations (K threads)
  python pretrain.py --parallel --threads 8 --generations 100
  
  # Custom aspect ratios and food counts
  python pretrain.py --parallel --aspect-ratios 16:9 4:3 21:9 1:1 --food-range 1 2 3
  
  # Visualize training in real-time (single mode only)
  python pretrain.py -v
  
  # More generations and fish
  python pretrain.py --generations 500 --fish 200
  
  # All together (parallel)
  python pretrain.py --parallel --threads 12 --generations 200 --fish 150
        """
    )
    parser.add_argument('-g', '--generations', type=int, default=100, 
                       help='Number of generations (default: 100)')
    parser.add_argument('--fish', type=int, default=100, 
                       help='Number of fish (default: 100)')
    parser.add_argument('--foods', type=int, default=3, 
                       help='Number of food sources for single mode (default: 3)')
    parser.add_argument('--gen-time', type=float, default=60.0, 
                       help='Generation time in seconds (default: 60.0)')
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='Show real-time visualization (single mode only, requires matplotlib)')
    parser.add_argument('-o', '--output', type=str, default='pretrained_weights.json', 
                       help='Output file for single mode (default: pretrained_weights.json)')
    parser.add_argument('--n-brains', type=int, default=20, 
                       help='Number of best brains to save (default: 20)')
    
    # Parallel training options
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel training with multiple variants')
    parser.add_argument('--threads', type=int, default=None,
                       help='Number of parallel threads (default: CPU count)')
    parser.add_argument('--aspect-ratios', type=str, nargs='+', 
                       default=['16:9', '4:3', '21:9', '1:1'],
                       help='Aspect ratios to test (format: W:H), e.g. 16:9 4:3 (default: 16:9 4:3 21:9 1:1)')
    parser.add_argument('--food-range', type=int, nargs='+', default=[1, 2, 3],
                       help='Food counts to test (default: 1 2 3)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for parallel mode (default: current directory)')
    
    args = parser.parse_args()
    
    if args.parallel:
        # ─── PARALLEL MODE ───
        # Parse aspect ratios
        aspect_ratios = []
        for ar_str in args.aspect_ratios:
            try:
                w, h = map(int, ar_str.split(':'))
                aspect_ratios.append((w, h))
            except:
                print(f"Warning: Invalid aspect ratio '{ar_str}', skipping")
        
        if not aspect_ratios:
            print("Error: No valid aspect ratios specified")
            return
        
        # Configure base simulation
        base_config = Config.from_file(
            fish_count=args.fish,
            generation_time=args.gen_time
        )
        
        # Run parallel training
        results = train_parallel(
            generations=args.generations,
            base_config=base_config,
            aspect_ratios=aspect_ratios,
            food_counts=args.food_range,
            n_threads=args.threads,
            output_dir=args.output_dir
        )
        
        print("\n🎉 Parallel training complete! Next steps:")
        print(f"  1. Review {args.output_dir}/training_summary.json for detailed statistics")
        print(f"  2. Copy {args.output_dir}/pretrained_weights.json to assets/")
        print(f"  3. Reload your website to see the evolved fish!")
        print(f"\nThe pretrained_weights.json contains the top 20 brains from across all variants.")
        
    else:
        # ─── SINGLE MODE ───
        # Configure simulation (load from file with overrides)
        config = Config.from_file(
            fish_count=args.fish,
            food_count=args.foods,
            generation_time=args.gen_time
        )
        
        # Train
        sim = train(
            generations=args.generations,
            config=config,
            verbose=True,
            visualize=args.visualize
        )
        
        # Save
        save_weights(sim, args.output, args.n_brains)
        
        print("🎉 Pre-training complete! Next steps:")
        print(f"  1. cp {args.output} assets/")
        print(f"  2. Reload your website")
        print(f"  3. Watch evolution continue from generation {sim.generation}!")


if __name__ == '__main__':
    main()
