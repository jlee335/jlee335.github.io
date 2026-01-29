import simpy
import torch
import torch.nn as nn
import numpy as np
import json
import copy
from concurrent.futures import ProcessPoolExecutor

# Load Configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

class AgentBrain(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Input increased to 8 (added 2 wall sensors)
        self.l1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(32, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return self.tanh(x)

    def mutate(self, rate=0.05):
        with torch.no_grad():
            for param in self.parameters():
                if np.random.rand() < rate:
                    param.add_(torch.randn_like(param) * 0.1)

class Simulation:
    def __init__(self, brain_weights):
        self.params = copy.deepcopy(CONFIG['simulation'])
        self.agents_cfg = CONFIG['agents']
        self.aspect_ratios = [
            r for r in self.params.get('aspect_ratios', [self.params['width'] / self.params['height']])
            if r > 0
        ]
        self._set_aspect_ratio()
        min_dim = min(self.params['width'], self.params['height'])
        fish_radius_value = self.agents_cfg.get('fish_radius', 0.013)
        predator_radius_value = self.agents_cfg.get('predator_radius', 0.02)
        adjacent_radius_value = self.agents_cfg.get('adjacent_radius', 0.083)
        self.fish_radius = fish_radius_value * min_dim if fish_radius_value <= 1 else fish_radius_value
        self.predator_radius = predator_radius_value * min_dim if predator_radius_value <= 1 else predator_radius_value
        self.predator_stun_seconds = self.agents_cfg.get('predator_stun_seconds', 1.0)
        self.edge_eat_margin = self.agents_cfg.get('edge_eat_margin', 30.0)
        self.adjacent_radius = adjacent_radius_value * min_dim if adjacent_radius_value <= 1 else adjacent_radius_value
        self.adjacent_max = self.agents_cfg.get('adjacent_max', 6)
        self.speed_scale = self.agents_cfg.get('speed_scale', self.agents_cfg.get('fish_speed', 4.0) * 1.5)
        self.base_eat_prob = self.agents_cfg.get('base_eat_prob', 1.0)
        self.density_weight = self.agents_cfg.get('density_weight', 0.6)
        self.speed_weight = self.agents_cfg.get('speed_weight', 0.4)
        
        # Inputs: [Ally_dx, Ally_dy, Pred_dx, Pred_dy, Vx, Vy, Wall_x, Wall_y]
        self.brain = AgentBrain(8, 3) 
        if brain_weights:
            self.brain.load_state_dict(brain_weights)
        
        self.agents = []
        self.predators = []
        self.score = 0
        
        # Spawn in center area to avoid instant wall collision
        fish_margin = max(
            0,
            min(
                100,
                (self.params['width'] - self.fish_radius * 2) / 2,
                (self.params['height'] - self.fish_radius * 2) / 2
            )
        )
        predator_margin = max(
            0,
            min(
                100,
                (self.params['width'] - self.predator_radius * 2) / 2,
                (self.params['height'] - self.predator_radius * 2) / 2
            )
        )
        for _ in range(self.agents_cfg['fish_count']):
            self.agents.append(self._random_entity(fish_margin))
        for _ in range(self.agents_cfg['predator_count']):
            self.predators.append(self._random_entity(predator_margin))

    def _set_aspect_ratio(self):
        if not self.aspect_ratios:
            return
        ratio = float(np.random.choice(self.aspect_ratios))
        area = float(self.params['width'] * self.params['height'])
        self.params['width'] = float(np.sqrt(area * ratio))
        self.params['height'] = float(np.sqrt(area / ratio))

    def _random_entity(self, margin):
        return {
            'x': np.random.uniform(margin, self.params['width'] - margin),
            'y': np.random.uniform(margin, self.params['height'] - margin),
            'vx': np.random.randn(), 'vy': np.random.randn(),
            'alive': True,
            'stun': 0.0
        }

    def get_sensor_data(self, agent):
        # 1. Ally & Predator Sensing (Same as before)
        nearest_ally_d = 9999
        ally_vec = [0, 0]
        for other in self.agents:
            if other is agent or not other['alive']: continue
            dx = (other['x'] - agent['x']) / 100.0
            dy = (other['y'] - agent['y']) / 100.0
            dist = dx*dx + dy*dy
            if dist < nearest_ally_d:
                nearest_ally_d = dist
                ally_vec = [dx, dy]

        nearest_pred_d = 9999
        pred_vec = [0, 0]
        for pred in self.predators:
            dx = (pred['x'] - agent['x']) / 100.0
            dy = (pred['y'] - agent['y']) / 100.0
            dist = dx*dx + dy*dy
            if dist < nearest_pred_d:
                nearest_pred_d = dist
                pred_vec = [dx, dy]

        # 2. Wall Sensing (NEW)
        # Normalize position to -1 (Left/Top) to 1 (Right/Bottom)
        wall_x = (agent['x'] / self.params['width']) * 2 - 1
        wall_y = (agent['y'] / self.params['height']) * 2 - 1

        state = np.array([
            ally_vec[0], ally_vec[1],
            pred_vec[0], pred_vec[1],
            agent['vx'], agent['vy'],
            wall_x, wall_y  # <--- The agents now "know" where they are
        ], dtype=np.float32)
        
        return torch.FloatTensor(state)

    def apply_wall_repulsion(self, entity, force=0.5):
        # Soft Repulsion: Pushes them away if they get too close to edges
        margin = 50
        if entity['x'] < margin: entity['vx'] += force
        if entity['x'] > self.params['width'] - margin: entity['vx'] -= force
        if entity['y'] < margin: entity['vy'] += force
        if entity['y'] > self.params['height'] - margin: entity['vy'] -= force
        
        # Hard Clamp (prevent leaving screen)
        entity['x'] = np.clip(entity['x'], 0, self.params['width'])
        entity['y'] = np.clip(entity['y'], 0, self.params['height'])

    def resolve_collisions(self, entities, radius, push=0.2):
        for i in range(len(entities)):
            a = entities[i]
            if a.get('alive') is False:
                continue
            for j in range(i + 1, len(entities)):
                b = entities[j]
                if b.get('alive') is False:
                    continue
                dx = b['x'] - a['x']
                dy = b['y'] - a['y']
                min_dist = radius * 2
                dist_sq = dx * dx + dy * dy
                if 0 < dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    a['x'] -= nx * overlap * 0.5
                    a['y'] -= ny * overlap * 0.5
                    b['x'] += nx * overlap * 0.5
                    b['y'] += ny * overlap * 0.5
                    a['vx'] -= nx * push
                    a['vy'] -= ny * push
                    b['vx'] += nx * push
                    b['vy'] += ny * push

    def escape_chance(self, agent):
        neighbors = 0
        radius_sq = self.adjacent_radius * self.adjacent_radius
        for other in self.agents:
            if other is agent or not other['alive']:
                continue
            dx = other['x'] - agent['x']
            dy = other['y'] - agent['y']
            if dx * dx + dy * dy <= radius_sq:
                neighbors += 1
        speed = np.sqrt(agent['vx'] ** 2 + agent['vy'] ** 2)
        density_factor = min(neighbors / self.adjacent_max, 1.0)
        speed_factor = min(speed / self.speed_scale, 1.0)
        escape = density_factor * self.density_weight + speed_factor * self.speed_weight
        return min(max(escape, 0.0), 0.9)

    def is_near_edge(self, agent):
        margin = max(0.0, self.edge_eat_margin) + self.fish_radius
        return (
            agent['x'] <= margin or
            agent['x'] >= self.params['width'] - margin or
            agent['y'] <= margin or
            agent['y'] >= self.params['height'] - margin
        )

    def step(self):
        # Update Predators (Chase NEAREST, not center)
        for p in self.predators:
            best_target = None
            min_dist = 99999

            if p.get('stun', 0.0) > 0.0:
                p['stun'] = max(0.0, p['stun'] - self.params.get('dt', 0.1))
                p['vx'] *= 0.9
                p['vy'] *= 0.9
            else:
                for a in self.agents:
                    if a['alive']:
                        d = (a['x']-p['x'])**2 + (a['y']-p['y'])**2
                        if d < min_dist:
                            min_dist = d
                            best_target = a
                
                if best_target:
                    dx = best_target['x'] - p['x']
                    dy = best_target['y'] - p['y']
                    mag = np.sqrt(dx**2 + dy**2) + 1e-5
                    p['vx'] += (dx/mag) * 0.25
                    p['vy'] += (dy/mag) * 0.25
            
            # Friction & Walls
            speed = np.sqrt(p['vx']**2 + p['vy']**2)
            if speed > self.agents_cfg['predator_speed']:
                p['vx'] = (p['vx']/speed) * self.agents_cfg['predator_speed']
                p['vy'] = (p['vy']/speed) * self.agents_cfg['predator_speed']
            
            p['x'] += p['vx']; p['y'] += p['vy']
            self.apply_wall_repulsion(p, force=0.2) # Match prey reachability

        self.resolve_collisions(self.predators, self.predator_radius, push=0.3)
        for p in self.predators:
            self.apply_wall_repulsion(p, force=0.0)

        # Update Agents
        alive_count = 0
        for agent in self.agents:
            if not agent['alive']: continue
            alive_count += 1
            
            # Brain
            state = self.get_sensor_data(agent)
            with torch.no_grad():
                action = self.brain(state).numpy()

            # Physics
            agent['vx'] += action[0] * 0.5
            agent['vy'] += action[1] * 0.5
            
            # Add explicit wall avoidance reward implicitly via survival
            # But here we enforce physics so they don't get stuck
            self.apply_wall_repulsion(agent, force=0.2)

            agent['vx'] *= 0.95
            agent['vy'] *= 0.95
            
            agent['x'] += agent['vx']
            agent['y'] += agent['vy']
            
            # Collision
            for p in self.predators:
                if (agent['x']-p['x'])**2 + (agent['y']-p['y'])**2 < (self.fish_radius + self.predator_radius) ** 2:
                    if p.get('stun', 0.0) > 0.0:
                        continue
                    eaten_prob_base = self.base_eat_prob * (1 - self.escape_chance(agent))
                    eaten_prob = 1.0 if self.is_near_edge(agent) else min(1.0, eaten_prob_base)
                    if np.random.rand() < eaten_prob:
                        agent['alive'] = False
                        self.score -= 50
                    else:
                        p['stun'] = max(p.get('stun', 0.0), self.predator_stun_seconds)
                        dx = agent['x'] - p['x']
                        dy = agent['y'] - p['y']
                        mag = np.sqrt(dx * dx + dy * dy) + 1e-6
                        agent['vx'] += (dx / mag) * 1.2
                        agent['vy'] += (dy / mag) * 1.2

        self.resolve_collisions(self.agents, self.fish_radius, push=0.2)
        for agent in self.agents:
            if agent['alive']:
                self.apply_wall_repulsion(agent, force=0.0)

        self.score += alive_count * 0.1

    def run(self, steps=300):
        for _ in range(steps):
            self.step()
        return self.score

def save_weights_to_json(model, filename="weights.json"):
    state_dict = model.state_dict()
    data = {
        "l1_w": state_dict['l1.weight'].tolist(),
        "l1_b": state_dict['l1.bias'].tolist(),
        "l2_w": state_dict['l2.weight'].tolist(),
        "l2_b": state_dict['l2.bias'].tolist(),
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def evaluate(weights):
    sim = Simulation(weights)
    return sim.run()

if __name__ == '__main__':
    # Increase mutation rate slightly to help them learn the new wall sensors
    pop_size = CONFIG['evolution']['population_size']
    population = [AgentBrain(8, 3) for _ in range(pop_size)]
    
    print(f"Evolving {CONFIG['evolution']['generations']} generations (Wall-Aware Mode)...")
    
    for gen in range(CONFIG['evolution']['generations']):
        weights_list = [p.state_dict() for p in population]
        with ProcessPoolExecutor() as executor:
            scores = list(executor.map(evaluate, weights_list))
        
        sorted_indices = np.argsort(scores)[::-1]
        print(f"Gen {gen}: Score {scores[sorted_indices[0]]:.2f}")
        
        top_k = int(pop_size * 0.2)
        elites = [population[i] for i in sorted_indices[:top_k]]
        next_gen = []
        while len(next_gen) < pop_size:
            parent = np.random.choice(elites)
            child = copy.deepcopy(parent)
            child.mutate()
            next_gen.append(child)
        population = next_gen

    save_weights_to_json(elites[0])
    print("Done.")