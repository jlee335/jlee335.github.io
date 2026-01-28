/**
 * Fish foraging simulation with evolved neural-network controllers
 * ---------------------------------------------------------------
 *
 * OVERVIEW
 *   - N fish (see FISH_COUNT) each have a 2-layer feedforward NN (tanh hidden).
 *   - Fish have LIMITED vision (1/10 of max screen dimension) and use ANTENNA communication.
 *   - Fish broadcast 3-value antenna signals to neighbors, enabling emergent communication.
 *   - Fish can also sense nearest neighbor's position and heading for flocking behavior.
 *   - Fitness = food eaten. Every 30s, evolved via tournament selection with crossover.
 *
 * FOOD
 *   - One food target at a time. A single autonomous pellet moves at constant speed
 *     and bounces off walls; when a fish touches it, the pellet collides like a ball
 *     instead of respawning.
 *
 * FISH BRAIN (per fish)
 *   - Inputs (11): [ cos(angle_to_food), sin(angle_to_food), distNorm,
 *                    cos(angle_to_neighbor), sin(angle_to_neighbor), cos(neighbor_heading), neighbor_dist,
 *                    last_action, antenna_in[3] ].
 *     Food target can be: (1) directly seen, or (2) default center.
 *     Antenna inputs receive 3-value signals from neighbors.
 *     Neighbor inputs enable flocking (cohesion, separation, alignment).
 *   - Hidden: 8 units, tanh. Output: 6 units.
 *   - Action each tick: argmax(output[0:3]) → 0 = forward only, 1 = turn 30° left + forward, 2 = turn 30° right + forward.
 *   - Antenna output: tanh(output[3:6]) → 3 values broadcast to neighbors within comm range.
 *     “Forward” adds an impulse in current heading; physics integrates velocity every frame.
 *
 * COMMUNICATION SYSTEM
 *   - Visual range: configurable factor of max(width, height) for food spotting (default: 0.1).
 *   - Communication range: separate configurable factor for fish-to-fish comms (default: 0.15).
 *   - Each fish outputs 3-value antenna signal, broadcast to neighbors if magnitude > threshold.
 *   - Signal threshold: fish must learn to output strong signals (magnitude > 0.5) to communicate.
 *   - If multiple signals arrive, only the latest is used (last one in iteration wins).
 *   - Signal is consumed after being read (cleared each brain tick).
 *   - Visualization: 3 signal values map to RGB color (dimmed), stronger signals = brighter/thicker lines.
 *
 * PHYSICS
 *   - Fish and (when active) the pellet have vx, vy; position += velocity*dt each frame.
 *   - Damping and max speed applied; walls reflect velocity and update fish heading.
 *
 * RENDERING
 *   - Canvas id must be "neural-canvas". Top-left HUD: generation number and full
 *     leaderboard (food eaten). Top 5 fish drawn with a highlight (glow).
 *   - Color-coded lines between fish show active antenna signal transmission.
 *   - Line color = RGB mapped from antenna signal values (dimmed for subtlety).
 *
 * FOR AI AGENTS
 *   - Config: FISH_COUNT, TICK_MS, FISH_SPEED, DAMPING, MAX_SPEED, INPUT/HIDDEN/OUTPUT_SIZE,
 *     VISUAL_RANGE_FACTOR, COMM_RANGE_FACTOR, FOOD_SPEED, MUTATION_RATE, MUTATION_STRENGTH.
 *   - State: fish[], food, generation, commLinks[], visualRange, commRange.
 *   - Entry: resize(), init(), animate().
 */
const canvas = document.getElementById('neural-canvas');
const ctx = canvas ? canvas.getContext('2d') : null;
const toggleButton = document.getElementById('background-toggle');
const profileImage = document.getElementById('profile-image');

let width, height;

// ─── Config (loaded from assets/config.json) ────────────────────────────
let CONFIG = null;
let FISH_COUNT, TICK_MS, FISH_SPEED, DAMPING, MAX_SPEED;
let TURN_DEG, TURN_RAD, FISH_RADIUS, FOOD_RADIUS, EAT_DIST;
let INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, FOOD_SPEED;
let MUTATION_RATE, MUTATION_STRENGTH;
let COMM_DELAY_MS, VISUAL_RANGE_FACTOR, COMM_RANGE_FACTOR;
let FOV_DEG, FOV_RAD;
let SIGNAL_THRESHOLD; // Minimum signal magnitude to broadcast

// ─── State ─────────────────────────────────────────────────────────────
let fish = [];
let food = null;
let lastTick = 0;
let lastFrameTime = 0;
let generation = 1;
let commLinks = [];              // Active antenna communication links for visualization
let maxDim = 1;                  // Maximum dimension (width or height)
let visualRange = 0;             // Dynamic visual range for spotting food
let commRange = 0;               // Dynamic communication range for fish-to-fish
let simulationEnabled = false;
let simulationInitialized = false;
let animationFrameId = null;
let startToken = 0;

const FOOD_RESTITUTION = 0.9;    // Bounciness for fish-food collisions

function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    maxDim = Math.max(width, height);
    if (VISUAL_RANGE_FACTOR && COMM_RANGE_FACTOR) {
        visualRange = maxDim * VISUAL_RANGE_FACTOR;
        commRange = maxDim * COMM_RANGE_FACTOR;
    } else {
        visualRange = 0;
        commRange = 0;
    }
    if (canvas) {
        canvas.width = width;
        canvas.height = height;
    }
}

function getFoodTarget() {
    return food ? { x: food.x, y: food.y } : { x: width / 2, y: height / 2 };
}

// ─── Neural Net (2-layer: INPUT_SIZE → HIDDEN_SIZE → OUTPUT_SIZE, tanh) ─
class NeuralNet {
    constructor() {
        this.W1 = randMatrix(HIDDEN_SIZE, INPUT_SIZE + 1);
        this.W2 = randMatrix(OUTPUT_SIZE, HIDDEN_SIZE + 1);
    }

    static fromParent(parent) {
        const nn = new NeuralNet();
        nn.W1 = parent.W1.map(row => row.map(w => mutate(w)));
        nn.W2 = parent.W2.map(row => row.map(w => mutate(w)));
    }

    forward(inputs) {
        const withBias = [...inputs, 1];
        const hidden = this.W1.map(row =>
            Math.tanh(dot(row, withBias))
        );
        const withBiasH = [...hidden, 1];
        const out = this.W2.map(row => dot(row, withBiasH));
        return out;
    }

    act(inputs) {
        const out = this.forward(inputs);
        // First 3 outputs are actions
        const actions = out.slice(0, 3);
        const action = actions.indexOf(Math.max(...actions));
        // Last 3 outputs are antenna signal (use tanh to bound between -1 and 1)
        const antenna = out.slice(3, 6).map(x => Math.tanh(x));
        return { action, antenna };
    }
}

function randMatrix(rows, cols) {
    return Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => (Math.random() - 0.5) * 2)
    );
}

function dot(a, b) {
    return a.reduce((s, x, i) => s + x * b[i], 0);
}

function mutate(w) {
    if (Math.random() < MUTATION_RATE) {
        return w + (Math.random() - 0.5) * 2 * MUTATION_STRENGTH;
    }
    return w;
}

// ─── Fish ───────────────────────────────────────────────────────────────
class Fish {
    constructor(brain = null) {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = 0;
        this.vy = 0;
        this.angle = Math.random() * Math.PI * 2;
        this.brain = brain || new NeuralNet();
        this.foodEaten = 0;
        
        // Antenna communication system
        this.antennaOut = [0, 0, 0];  // Signal to broadcast
        this.antennaIn = [0, 0, 0];   // Latest received signal (consumed after use)
        
        // Proprioception (memory of previous action)
        this.lastAction = 0;          // 0=forward, 1=left, 2=right
    }

    getInputs() {
        // --- 1. Food Inputs ---
        const actualTarget = getFoodTarget();
        const dx = actualTarget.x - this.x;
        const dy = actualTarget.y - this.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        
        let target = null;
        const canSeeFood = d <= visualRange;
        
        if (canSeeFood) {
            // Can directly see the food
            target = actualTarget;
        } else {
            // No information available - default to center
            target = { x: width / 2, y: height / 2 };
        }
        
        const tdx = target.x - this.x;
        const tdy = target.y - this.y;
        const td = Math.sqrt(tdx * tdx + tdy * tdy);
        
        // Normalize distance by max dimension to prevent overfitting to canvas size
        const distNorm = Math.min(1, td / maxDim);
        const angleToFood = Math.atan2(tdy, tdx);
        const relAngle = angleToFood - this.angle;
        const cosRel = Math.cos(relAngle);
        const sinRel = Math.sin(relAngle);
        
        const foodInputs = [cosRel, sinRel, distNorm];
        
        // --- 2. Neighbor Inputs (Social positioning) ---
        let closestNeighbor = null;
        let minNDist = Infinity;
        const commRangeSq = commRange * commRange;
        
        for (const neighbor of fish) {
            if (neighbor === this) continue;
            
            const ndx = neighbor.x - this.x;
            const ndy = neighbor.y - this.y;
            const d2 = ndx * ndx + ndy * ndy;
            
            if (d2 < commRangeSq && d2 < minNDist) {
                minNDist = d2;
                closestNeighbor = neighbor;
            }
        }
        
        let socialInputs;
        if (closestNeighbor) {
            // Relative Angle to neighbor (Where are they?)
            const nx = closestNeighbor.x - this.x;
            const ny = closestNeighbor.y - this.y;
            const angleToNeighbor = Math.atan2(ny, nx) - this.angle;
            
            // Relative Heading of neighbor (Where are they looking?)
            const headingDiff = closestNeighbor.angle - this.angle;
            
            const neighborDistNorm = Math.sqrt(minNDist) / maxDim;
            socialInputs = [
                Math.cos(angleToNeighbor),
                Math.sin(angleToNeighbor),
                Math.cos(headingDiff),  // Alignment
                neighborDistNorm
            ];
        } else {
            socialInputs = [0.0, 0.0, 0.0, 1.0];
        }
        
        // --- 3. Proprioception (Memory) ---
        const memoryInput = [this.lastAction - 1];  // -1, 0, or 1
        
        // --- 4. Antenna Input (Received signal) ---
        const antennaInputs = [...this.antennaIn];
        
        // Combine: 3 (Food) + 4 (Social) + 1 (Memory) + 3 (Antenna) = 11 inputs
        return [...foodInputs, ...socialInputs, ...memoryInput, ...antennaInputs];
    }

    push() {
        const inputs = this.getInputs();
        const { action, antenna } = this.brain.act(inputs);
        if (action === 1) this.angle -= TURN_RAD;
        else if (action === 2) this.angle += TURN_RAD;
        this.vx += Math.cos(this.angle) * FISH_SPEED;
        this.vy += Math.sin(this.angle) * FISH_SPEED;
        
        // Store action for next tick (proprioception)
        this.lastAction = action;
        
        // Broadcast antenna signal (fish learns to output strong signals to communicate)
        this.antennaOut = antenna;
        
        // Clear received signal after consumption
        this.antennaIn = [0, 0, 0];
    }
    
    getSignalMagnitude() {
        // Calculate L2 norm of antenna output
        return Math.sqrt(
            this.antennaOut[0] * this.antennaOut[0] +
            this.antennaOut[1] * this.antennaOut[1] +
            this.antennaOut[2] * this.antennaOut[2]
        );
    }

    physics(dt) {
        this.x += this.vx * dt;
        this.y += this.vy * dt;
        this.vx *= DAMPING;
        this.vy *= DAMPING;
        const s = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
        if (s > MAX_SPEED) {
            this.vx = (this.vx / s) * MAX_SPEED;
            this.vy = (this.vy / s) * MAX_SPEED;
        }
        this.x = Math.max(0, Math.min(width, this.x));
        this.y = Math.max(0, Math.min(height, this.y));
        if (this.x <= 0 || this.x >= width) {
            this.vx = -this.vx;
            this.angle = Math.atan2(this.vy, this.vx);
        }
        if (this.y <= 0 || this.y >= height) {
            this.vy = -this.vy;
            this.angle = Math.atan2(this.vy, this.vx);
        }

        const target = getFoodTarget();
        const dx = target.x - this.x;
        const dy = target.y - this.y;
        if (dx * dx + dy * dy < EAT_DIST * EAT_DIST) {
            if (food) {
                resolveFoodCollision(this, food);
                // Only count if food hasn't been "eaten" this frame by another fish
                if (!food.eatenThisFrame) {
                    this.foodEaten++;
                    food.eatenThisFrame = true;
                }
            }
        }
    }

    draw(highlight = false) {
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);
        if (highlight) {
            ctx.shadowColor = 'rgba(120, 160, 200, 0.6)';
            ctx.shadowBlur = 8;
        }
        
        // Calculate information freshness for color
        const actualTarget = getFoodTarget();
        const dx = actualTarget.x - this.x;
        const dy = actualTarget.y - this.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        const canSeeFood = d <= visualRange;
        
        let alpha = 0.6; // Base opacity
        let saturation = 1.0; // Base saturation
        
        if (canSeeFood) {
            // Directly seeing food - strong, vibrant color
            alpha = 0.8;
            saturation = 1.2;
        } else {
            // Not seeing food - dim
            alpha = 0.4;
            saturation = 0.6;
        }
        
        // Apply color
        const baseR = 72 * saturation;
        const baseG = 120 * saturation;
        const baseB = 150 * saturation;
        ctx.fillStyle = `rgba(${baseR}, ${baseG}, ${baseB}, ${alpha})`;
        
        const strokeR = 48 * saturation;
        const strokeG = 80 * saturation;
        const strokeB = 100 * saturation;
        ctx.strokeStyle = `rgba(${strokeR}, ${strokeG}, ${strokeB}, ${alpha * 0.8})`;
        ctx.lineWidth = 0.5;
        
        ctx.beginPath();
        ctx.moveTo(FISH_RADIUS, 0);
        ctx.lineTo(-FISH_RADIUS * 0.8, FISH_RADIUS * 0.6);
        ctx.lineTo(-FISH_RADIUS * 0.6, 0);
        ctx.lineTo(-FISH_RADIUS * 0.8, -FISH_RADIUS * 0.6);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
    }
}

// ─── Food ───────────────────────────────────────────────────────────────
function respawnFood() {
    const margin = 40;
    const angle = Math.random() * Math.PI * 2;
    food = {
        x: margin + Math.random() * (width - 2 * margin),
        y: margin + Math.random() * (height - 2 * margin),
        vx: Math.cos(angle) * FOOD_SPEED,
        vy: Math.sin(angle) * FOOD_SPEED,
    };
}

function updateFoodPhysics(dt) {
    if (!food) return;
    food.x += food.vx * dt;
    food.y += food.vy * dt;
    food.x = Math.max(0, Math.min(width, food.x));
    food.y = Math.max(0, Math.min(height, food.y));
    if (food.x <= 0 || food.x >= width) food.vx = -food.vx;
    if (food.y <= 0 || food.y >= height) food.vy = -food.vy;
}

function resolveFoodCollision(f, pellet) {
    const dx = pellet.x - f.x;
    const dy = pellet.y - f.y;
    const distSq = dx * dx + dy * dy;
    const minDist = EAT_DIST;
    if (distSq >= minDist * minDist) return;

    const dist = Math.sqrt(distSq) || 0.0001;
    const nx = dx / dist;
    const ny = dy / dist;

    // Push apart to avoid overlap
    const overlap = minDist - dist;
    f.x -= nx * overlap * 0.5;
    f.y -= ny * overlap * 0.5;
    pellet.x += nx * overlap * 0.5;
    pellet.y += ny * overlap * 0.5;

    // Elastic collision (equal mass)
    const rvx = pellet.vx - f.vx;
    const rvy = pellet.vy - f.vy;
    const velAlongNormal = rvx * nx + rvy * ny;
    if (velAlongNormal > 0) return;

    const impulse = -(1 + FOOD_RESTITUTION) * velAlongNormal / 2;
    f.vx -= impulse * nx;
    f.vy -= impulse * ny;
    pellet.vx += impulse * nx;
    pellet.vy += impulse * ny;
}

// ─── Evolution ──────────────────────────────────────────────────────────
function evolve() {
    fish.sort((a, b) => b.foodEaten - a.foodEaten);
    const keep = Math.max(1, Math.floor(FISH_COUNT / 2));
    const survivors = fish.slice(0, keep);
    const newFish = [];
    for (let i = 0; i < FISH_COUNT; i++) {
        const parent = survivors[i % survivors.length];
        const f = new Fish(NeuralNet.fromParent(parent.brain));
        f.x = Math.random() * width;
        f.y = Math.random() * height;
        f.vx = 0;
        f.vy = 0;
        f.angle = Math.random() * Math.PI * 2;
        newFish.push(f);
    }
    fish = newFish;
    generation++;
}

// ─── Init ──────────────────────────────────────────────────────────────
async function loadConfig() {
    try {
        const response = await fetch('assets/config.json');
        if (!response.ok) throw new Error('Config not found');
        const config = await response.json();
        
        // Neural network
        INPUT_SIZE = config.neural_network.input_size;
        HIDDEN_SIZE = config.neural_network.hidden_size;
        OUTPUT_SIZE = config.neural_network.output_size;
        
        // Physics
        FISH_SPEED = config.physics.fish_speed;
        DAMPING = config.physics.damping;
        MAX_SPEED = config.physics.max_speed;
        TICK_MS = config.physics.tick_ms;
        
        // Fish
        FISH_RADIUS = config.fish.fish_radius;
        TURN_DEG = config.fish.turn_deg;
        TURN_RAD = (TURN_DEG * Math.PI) / 180;
        VISUAL_RANGE_FACTOR = config.fish.visual_range_factor;
        COMM_RANGE_FACTOR = config.fish.comm_range_factor || config.fish.visual_range_factor;
        FOV_DEG = 30;
        FOV_RAD = (FOV_DEG * Math.PI) / 180;
        
        // Food
        FOOD_RADIUS = config.food.food_radius;
        FOOD_SPEED = config.food.food_speed;
        EAT_DIST = FISH_RADIUS + FOOD_RADIUS;
        
        // Communication
        COMM_DELAY_MS = config.communication.comm_delay_ms;
        SIGNAL_THRESHOLD = config.communication.signal_threshold || 0.5;
        
        // Evolution
        MUTATION_RATE = config.evolution.mutation_rate;
        MUTATION_STRENGTH = config.evolution.mutation_strength;
        
        // Simulation
        FISH_COUNT = config.simulation.fish_count;
        
        CONFIG = config;
        console.log('Configuration loaded from assets/config.json');
        return config;
    } catch (e) {
        console.error('Failed to load config:', e);
        throw e;
    }
}

async function loadPretrainedWeights() {
    try {
        const response = await fetch('assets/pretrained_weights.json');
        if (!response.ok) return null;
        const data = await response.json();
        console.log(`Loaded ${data.brains.length} pre-trained brains from generation ${data.generation}`);
        return data;
    } catch (e) {
        console.log('No pre-trained weights found, starting from scratch');
        return null;
    }
}

async function init() {
    // Load configuration first
    await loadConfig();
    resize();
    
    fish = [];
    
    // Try to load pre-trained weights
    const pretrained = await loadPretrainedWeights();
    
    if (pretrained && pretrained.brains.length > 0) {
        // Load pre-trained brains
        generation = pretrained.generation;
        for (let i = 0; i < FISH_COUNT; i++) {
            const brainIndex = Math.floor(Math.random() * pretrained.brains.length);
            const brainData = pretrained.brains[brainIndex];
            const brain = new NeuralNet();
            brain.W1 = brainData.W1;
            brain.W2 = brainData.W2;
            fish.push(new Fish(brain));
        }
        console.log(`Initialized ${FISH_COUNT} fish with pre-trained brains`);
    } else {
        // Random initialization
        for (let i = 0; i < FISH_COUNT; i++) {
            fish.push(new Fish());
        }
        console.log('Initialized with random brains');
    }
    
    respawnFood();
    lastTick = performance.now();
}

// ─── Antenna Communication System ──────────────────────────────────────
function updateAntennaCommunication() {
    const commRangeSq = commRange * commRange;
    const now = performance.now();
    
    // Clear old links
    commLinks = [];
    
    // Propagate antenna signals to neighbors (latest signal only)
    fish.forEach(receiver => {
        let latestSignal = null;
        let latestBroadcaster = null;
        let latestMagnitude = 0;
        
        // Find latest signal from neighbors within comm range
        // Last one in iteration wins (simple "latest" implementation)
        fish.forEach(broadcaster => {
            if (broadcaster === receiver) return;
            
            const dx = broadcaster.x - receiver.x;
            const dy = broadcaster.y - receiver.y;
            const d2 = dx * dx + dy * dy;
            
            if (d2 < commRangeSq) {
                // Check if signal magnitude exceeds threshold
                const magnitude = broadcaster.getSignalMagnitude();
                
                if (magnitude >= SIGNAL_THRESHOLD) {
                    // Within range AND strong enough signal - use this signal (last one wins)
                    latestSignal = [...broadcaster.antennaOut];
                    latestBroadcaster = broadcaster;
                    latestMagnitude = magnitude;
                }
            }
        });
        
        // Set received signal (if any)
        if (latestSignal) {
            receiver.antennaIn = latestSignal;
            
            // Add visual link for visualization (with magnitude and signal values for color)
            commLinks.push({
                from: latestBroadcaster,
                to: receiver,
                timestamp: now,
                age: 0,
                magnitude: latestMagnitude,
                signal: latestSignal  // Store signal for RGB coloring
            });
        }
    });
}

// ─── Tick (run brains every N ms; only apply push) ───────────────────────
function tick(now) {
    maybeEvolve(now);
    
    if (now - lastTick >= TICK_MS) {
        lastTick = now;
        // Step 1: All fish make decisions and broadcast antenna signals
        fish.forEach(f => f.push());
        // Step 2: Propagate antenna signals to neighbors
        updateAntennaCommunication();
    }
}

function updatePhysics(dt) {
    updateFoodPhysics(dt);
    
    // Mark food as not eaten this frame
    if (food) food.eatenThisFrame = false;
    
    // Update all fish physics
    fish.forEach(f => f.physics(dt));
    
    // No respawn on contact; food collides like a ball
}

// ─── Draw ──────────────────────────────────────────────────────────────
const TOP_N = 5;

function drawHUD(sorted) {
    const pad = 10;
    const lineHeight = 12;
    const fontSize = 10;
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillStyle = 'rgba(120, 140, 160, 0.55)';
    ctx.fillText(`Gen ${generation}`, pad, pad + fontSize);

    ctx.fillStyle = 'rgba(100, 120, 140, 0.5)';
    ctx.fillText('Leaderboard', pad, pad + lineHeight * 2 + fontSize);
    for (let i = 0; i < sorted.length; i++) {
        const f = sorted[i];
        ctx.fillStyle = i < TOP_N ? 'rgba(100, 120, 140, 0.65)' : 'rgba(100, 120, 140, 0.4)';
        ctx.fillText(`${i + 1}. ${f ? f.foodEaten : 0}`, pad, pad + lineHeight * (3 + i) + fontSize);
    }
}

function draw() {
    ctx.clearRect(0, 0, width, height);
    
    // Draw antenna communication links first (behind fish)
    drawAntennaLinks();

    const target = getFoodTarget();
    
    // Draw visual range indicator (very dim circle)
    ctx.strokeStyle = 'rgba(160, 90, 90, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(target.x, target.y, visualRange, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw food
    ctx.fillStyle = 'rgba(160, 90, 90, 0.6)';
    ctx.beginPath();
    ctx.arc(target.x, target.y, FOOD_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(130, 70, 70, 0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    const sorted = [...fish].sort((a, b) => b.foodEaten - a.foodEaten);
    const topFive = new Set(sorted.slice(0, TOP_N));
    fish.forEach(f => {
        f.draw(topFive.has(f));
    });
    drawHUD(sorted);
}

function drawAntennaLinks() {
    const now = performance.now();
    const linkLifetime = 300; // Links fade over 300ms
    
    // Update ages and filter old links
    commLinks = commLinks.map(link => ({
        ...link,
        age: now - link.timestamp
    })).filter(link => link.age < linkLifetime);
    
    // Draw each link
    commLinks.forEach(link => {
        const alpha = Math.max(0, 1 - link.age / linkLifetime);
        
        // Signal strength affects visual intensity (magnitude is 0 to ~1.7 for 3D vector)
        // Normalize to 0-1 range (max magnitude of unit vector in 3D is sqrt(3) ≈ 1.73)
        const strength = Math.min(1.0, link.magnitude / 1.73);
        
        // Map antenna signal values [-1, 1] to RGB [0, 255], but dimmed
        // Use signal values as RGB components
        const signal = link.signal;
        const r = Math.floor(((signal[0] + 1) / 2) * 255); // Map -1..1 to 0..255
        const g = Math.floor(((signal[1] + 1) / 2) * 255);
        const b = Math.floor(((signal[2] + 1) / 2) * 255);
        
        // Dimming factor: reduce intensity for subtler visualization
        const dimFactor = 0.4;
        const dimR = Math.floor(r * dimFactor);
        const dimG = Math.floor(g * dimFactor);
        const dimB = Math.floor(b * dimFactor);
        
        // Apply fading and strength modulation
        const lineAlpha = alpha * (0.3 + strength * 0.4);
        const lineWidth = 0.5 + strength * 0.5;
        
        // Draw line with RGB color based on antenna signal
        ctx.strokeStyle = `rgba(${dimR}, ${dimG}, ${dimB}, ${lineAlpha})`;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        ctx.moveTo(link.from.x, link.from.y);
        ctx.lineTo(link.to.x, link.to.y);
        ctx.stroke();
        
        // Draw a pulse along the line (slightly brighter than the line)
        const progress = (link.age / linkLifetime) * 0.5 + 0.25;
        const px = link.from.x + (link.to.x - link.from.x) * progress;
        const py = link.from.y + (link.to.y - link.from.y) * progress;
        const pulseSize = 2.0 + strength * 2.0;
        const pulseBrightness = 1.5; // Slightly brighter pulse
        ctx.fillStyle = `rgba(${Math.min(255, dimR * pulseBrightness)}, ${Math.min(255, dimG * pulseBrightness)}, ${Math.min(255, dimB * pulseBrightness)}, ${alpha * (0.5 + strength * 0.3)})`;
        ctx.beginPath();
        ctx.arc(px, py, pulseSize, 0, Math.PI * 2);
        ctx.fill();
    });
}

// ─── Animation loop ─────────────────────────────────────────────────────
function animate(now = 0) {
    if (!simulationEnabled) {
        animationFrameId = null;
        return;
    }
    if (!ctx || !width || !height) {
        animationFrameId = requestAnimationFrame(animate);
        return;
    }
    const dt = lastFrameTime ? (now - lastFrameTime) / 1000 : 1 / 60;
    lastFrameTime = now;
    updatePhysics(dt);
    tick(now);
    draw();
    animationFrameId = requestAnimationFrame(animate);
}

// Evolve every 30s: survival of the fittest
let nextEvolve = 30000;
function maybeEvolve(now) {
    if (now > nextEvolve) {
        nextEvolve = now + 30000;
        evolve();
    }
}

window.addEventListener('resize', resize);

async function startSimulation() {
    const token = ++startToken;
    if (!ctx || !canvas) return;
    if (!simulationInitialized) {
        await init();
        simulationInitialized = true;
    }
    if (token !== startToken) return;
    simulationEnabled = true;
    lastFrameTime = 0;
    lastTick = performance.now();
    animationFrameId = requestAnimationFrame(animate);
}

function stopSimulation() {
    startToken += 1;
    simulationEnabled = false;
    if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (ctx && width && height) {
        ctx.clearRect(0, 0, width, height);
    }
}

function setToggleState(enabled) {
    if (toggleButton) {
        toggleButton.setAttribute('aria-pressed', String(enabled));
    }
    if (profileImage) {
        profileImage.classList.toggle('is-dim', enabled);
    }
}

if (toggleButton) {
    toggleButton.addEventListener('click', () => {
        const next = !simulationEnabled;
        setToggleState(next);
        if (next) {
            startSimulation();
        } else {
            stopSimulation();
        }
    });
}

setToggleState(false);
