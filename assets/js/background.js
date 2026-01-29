(() => {
    'use strict';

    const canvas = document.getElementById('neural-canvas');
    if (!canvas) {
        return;
    }
    const ctx = canvas.getContext('2d');

    const statusEl = document.getElementById('status');
    const toggleButton = document.getElementById('background-toggle');
    const profileImage = document.getElementById('profile-image');

    const CONFIG_URL = 'test/config.json';
    const WEIGHTS_URL = 'test/weights.json';

    const DEFAULT_CONFIG = {
        simulation: {
            width: 800,
            height: 600,
            duration: 500,
            dt: 0.1,
            aspect_ratios: [0.75, 1.0, 1.33, 1.6]
        },
        agents: {
            fish_count: 30,
            predator_count: 3,
            fish_speed: 4.0,
            predator_speed: 6.5,
            fish_radius: 0.013,
            predator_radius: 0.02,
            predator_stun_seconds: 1.0,
            edge_eat_margin: 30.0,
            corner_radius_factor: 0.08,
            corner_radius_min: 24.0,
            corner_radius_max: 140.0,
            adjacent_radius: 0.083,
            adjacent_max: 6,
            speed_scale: 6.0,
            base_eat_prob: 1.0,
            density_weight: 0.6,
            speed_weight: 0.4,
            vision_radius: 0.17,
            vision_pixels: 60,
            comm_channels: 1
        },
        evolution: { generations: 20, population_size: 50, mutation_rate: 0.05 }
    };

    let config;
    let brain;
    let agents = [];
    let predators = [];
    let running = false;
    let userEnabled = false;
    let rafId = null;
    let lastTimestamp = null;
    const viewport = { width: 0, height: 0 };
    const simBounds = { x: 0, y: 0, width: 0, height: 0 };
    let FISH_RADIUS = 8;
    let PREDATOR_RADIUS = 12;
    let CORNER_RADIUS_FACTOR = 0.08;
    let CORNER_RADIUS_MIN = 24;
    let CORNER_RADIUS_MAX = 140;
    let ADJACENT_RADIUS = 50;
    let ADJACENT_MAX = 6;
    let SPEED_SCALE = 6.5;
    let BASE_EAT_PROB = 1.0;
    let DENSITY_WEIGHT = 0.6;
    let SPEED_WEIGHT = 0.4;
    let PREDATOR_STUN_SECONDS = 1.0;
    let EDGE_EAT_MARGIN = 30;
    let FISH_SPEED = 4.0;
    let FISH_RADIUS_RATIO = 0.013;
    let PREDATOR_RADIUS_RATIO = 0.02;
    let ADJACENT_RADIUS_RATIO = 0.083;
    let VISION_PIXELS = 60;
    let VISION_RADIUS = 100;
    let currentAspectRatio = null;

    class NeuralNet {
        constructor(weights) {
            this.w1 = weights.l1_w; // Shape [32, input_size]
            this.b1 = weights.l1_b; // Shape [32]
            this.w2 = weights.l2_w; // Shape [3, 32]
            this.b2 = weights.l2_b; // Shape [3]
        }

        dense(input, weights, bias) {
            let output = new Array(weights.length).fill(0);
            for (let i = 0; i < weights.length; i++) {
                let sum = 0;
                for (let j = 0; j < input.length; j++) {
                    sum += input[j] * weights[i][j];
                }
                output[i] = sum + bias[i];
            }
            return output;
        }

        relu(arr) { return arr.map(x => Math.max(0, x)); }
        tanh(arr) { return arr.map(x => Math.tanh(x)); }

        forward(inputs) {
            let h1 = this.dense(inputs, this.w1, this.b1);
            h1 = this.relu(h1);

            let out = this.dense(h1, this.w2, this.b2);
            return this.tanh(out); // [ForceX, ForceY, Color]
        }
    }

    function setStatus(text) {
        if (statusEl) {
            statusEl.textContent = text;
        }
    }

    async function fetchJsonWithFallback(url, fallback) {
        try {
            const res = await fetch(url);
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
            return await res.json();
        } catch (err) {
            console.warn(`Failed to load ${url}; using fallback.`, err);
            return fallback;
        }
    }

    function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        viewport.width = Math.max(1, window.innerWidth);
        viewport.height = Math.max(1, window.innerHeight);
        canvas.width = Math.floor(viewport.width * dpr);
        canvas.height = Math.floor(viewport.height * dpr);
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function setAspectRatioFromConfig() {
        if (config && config.simulation && Number.isFinite(config.simulation.width) && Number.isFinite(config.simulation.height)) {
            currentAspectRatio = config.simulation.width / config.simulation.height;
        } else {
            currentAspectRatio = viewport.width / viewport.height;
        }
    }

    function updateSimBounds() {
        const ratio = currentAspectRatio || viewport.width / viewport.height;
        let width = viewport.width;
        let height = width / ratio;
        if (height > viewport.height) {
            height = viewport.height;
            width = height * ratio;
        }
        simBounds.width = Math.max(1, width);
        simBounds.height = Math.max(1, height);
        simBounds.x = (viewport.width - simBounds.width) / 2;
        simBounds.y = (viewport.height - simBounds.height) / 2;
    }

    function updateScaledLengths() {
        const minDim = Math.max(1, Math.min(simBounds.width, simBounds.height));
        FISH_RADIUS = Math.max(2, FISH_RADIUS_RATIO * minDim);
        PREDATOR_RADIUS = Math.max(2, PREDATOR_RADIUS_RATIO * minDim);
        ADJACENT_RADIUS = Math.max(FISH_RADIUS * 2, ADJACENT_RADIUS_RATIO * minDim);
        agents.forEach(a => { a.radius = FISH_RADIUS; });
        predators.forEach(p => { p.radius = PREDATOR_RADIUS; });
    }

    function resetSimulation() {
        if (!config) {
            setStatus('Config not loaded');
            return;
        }
        setAspectRatioFromConfig();
        updateSimBounds();
        updateScaledLengths();
        agents = [];
        predators = [];
        const fishMargin = Math.max(
            0,
            Math.min(
                100,
                (simBounds.width - FISH_RADIUS * 2) / 2,
                (simBounds.height - FISH_RADIUS * 2) / 2
            )
        );
        const predatorMargin = Math.max(
            0,
            Math.min(
                100,
                (simBounds.width - PREDATOR_RADIUS * 2) / 2,
                (simBounds.height - PREDATOR_RADIUS * 2) / 2
            )
        );

        for (let i = 0; i < config.agents.fish_count; i++) {
            agents.push({
                x: simBounds.x + fishMargin + Math.random() * (simBounds.width - fishMargin * 2),
                y: simBounds.y + fishMargin + Math.random() * (simBounds.height - fishMargin * 2),
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                alive: true,
                color: 0,
                radius: FISH_RADIUS
            });
        }

        for (let i = 0; i < config.agents.predator_count; i++) {
            predators.push({
                x: simBounds.x + predatorMargin + Math.random() * (simBounds.width - predatorMargin * 2),
                y: simBounds.y + predatorMargin + Math.random() * (simBounds.height - predatorMargin * 2),
                vx: 0,
                vy: 0,
                radius: PREDATOR_RADIUS,
                stun: 0
            });
        }
    }

    function getCornerRadius() {
        const base = Math.min(simBounds.width, simBounds.height) * CORNER_RADIUS_FACTOR;
        return Math.max(CORNER_RADIUS_MIN, Math.min(CORNER_RADIUS_MAX, base));
    }

    function buildRoundedRectPath(radius) {
        const r = Math.max(0, Math.min(radius, simBounds.width / 2, simBounds.height / 2));
        ctx.beginPath();
        ctx.moveTo(simBounds.x + r, simBounds.y);
        ctx.lineTo(simBounds.x + simBounds.width - r, simBounds.y);
        ctx.arcTo(simBounds.x + simBounds.width, simBounds.y, simBounds.x + simBounds.width, simBounds.y + r, r);
        ctx.lineTo(simBounds.x + simBounds.width, simBounds.y + simBounds.height - r);
        ctx.arcTo(
            simBounds.x + simBounds.width,
            simBounds.y + simBounds.height,
            simBounds.x + simBounds.width - r,
            simBounds.y + simBounds.height,
            r
        );
        ctx.lineTo(simBounds.x + r, simBounds.y + simBounds.height);
        ctx.arcTo(simBounds.x, simBounds.y + simBounds.height, simBounds.x, simBounds.y + simBounds.height - r, r);
        ctx.lineTo(simBounds.x, simBounds.y + r);
        ctx.arcTo(simBounds.x, simBounds.y, simBounds.x + r, simBounds.y, r);
        ctx.closePath();
    }

    function getVisionBins(agent) {
        const enemyBins = new Array(VISION_PIXELS).fill(0);
        const friendlyBins = new Array(VISION_PIXELS).fill(0);
        const bestEnemyDistSq = new Array(VISION_PIXELS).fill(Infinity);
        const bestFriendlyDistSq = new Array(VISION_PIXELS).fill(Infinity);
        const twoPi = Math.PI * 2;
        const visionRadiusSq = VISION_RADIUS * VISION_RADIUS;

        const ingestEntity = (entity, isEnemy) => {
            const dx = entity.x - agent.x;
            const dy = entity.y - agent.y;
            const distSq = dx * dx + dy * dy;
            if (distSq > visionRadiusSq) {
                return;
            }
            let angle = Math.atan2(dy, dx);
            if (angle < 0) angle += twoPi;
            let idx = Math.floor((angle / twoPi) * VISION_PIXELS);
            if (idx >= VISION_PIXELS) idx = VISION_PIXELS - 1;
            if (isEnemy) {
                if (distSq < bestEnemyDistSq[idx]) {
                    bestEnemyDistSq[idx] = distSq;
                    enemyBins[idx] = Math.sqrt(distSq) / VISION_RADIUS;
                }
            } else if (distSq < bestFriendlyDistSq[idx]) {
                bestFriendlyDistSq[idx] = distSq;
                friendlyBins[idx] = Math.sqrt(distSq) / VISION_RADIUS;
            }
        };

        for (let other of agents) {
            if (other === agent || !other.alive) continue;
            ingestEntity(other, false);
        }

        for (let p of predators) {
            ingestEntity(p, true);
        }

        return { enemyBins, friendlyBins };
    }

    function getSensors(agent) {
        const { enemyBins, friendlyBins } = getVisionBins(agent);
        return enemyBins.concat(friendlyBins);
    }

    function clampToRoundedRect(entity) {
        const radius = entity.radius || 0;
        const cornerBase = getCornerRadius();
        const corner = Math.max(
            0,
            Math.min(
                cornerBase - radius,
                (simBounds.width - radius * 2) / 2,
                (simBounds.height - radius * 2) / 2
            )
        );

        entity.x = Math.min(Math.max(entity.x, simBounds.x + radius), simBounds.x + simBounds.width - radius);
        entity.y = Math.min(Math.max(entity.y, simBounds.y + radius), simBounds.y + simBounds.height - radius);

        if (corner <= 0) {
            return;
        }

        const corners = [
            { cx: simBounds.x + corner, cy: simBounds.y + corner },
            { cx: simBounds.x + simBounds.width - corner, cy: simBounds.y + corner },
            { cx: simBounds.x + corner, cy: simBounds.y + simBounds.height - corner },
            { cx: simBounds.x + simBounds.width - corner, cy: simBounds.y + simBounds.height - corner }
        ];

        corners.forEach(({ cx, cy }) => {
            const dx = entity.x - cx;
            const dy = entity.y - cy;
            if ((entity.x < simBounds.x + corner && cx === simBounds.x + corner) || (entity.x > simBounds.x + simBounds.width - corner && cx !== simBounds.x + corner)) {
                if ((entity.y < simBounds.y + corner && cy === simBounds.y + corner) || (entity.y > simBounds.y + simBounds.height - corner && cy !== simBounds.y + corner)) {
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist > corner) {
                        const scale = corner / (dist || 1);
                        entity.x = cx + dx * scale;
                        entity.y = cy + dy * scale;
                    }
                }
            }
        });
    }

    function applyWallRepulsion(entity, force) {
        const radius = entity.radius || 0;
        const maxInsetX = (simBounds.width - radius * 2) / 2;
        const maxInsetY = (simBounds.height - radius * 2) / 2;
        const margin = Math.max(0, Math.min(50, maxInsetX, maxInsetY));
        const beforeX = entity.x;
        const beforeY = entity.y;

        if (entity.x < simBounds.x + radius + margin) entity.vx += force;
        if (entity.x > simBounds.x + simBounds.width - radius - margin) entity.vx -= force;
        if (entity.y < simBounds.y + radius + margin) entity.vy += force;
        if (entity.y > simBounds.y + simBounds.height - radius - margin) entity.vy -= force;

        clampToRoundedRect(entity);

        const pushX = entity.x - beforeX;
        const pushY = entity.y - beforeY;
        if (pushX || pushY) {
            const mag = Math.sqrt(pushX * pushX + pushY * pushY) || 1;
            entity.vx += (pushX / mag) * force;
            entity.vy += (pushY / mag) * force;
        }
    }

    function resolveCollisions(entities, radius, push) {
        for (let i = 0; i < entities.length; i++) {
            const a = entities[i];
            if (a.alive === false) continue;
            for (let j = i + 1; j < entities.length; j++) {
                const b = entities[j];
                if (b.alive === false) continue;
                const dx = b.x - a.x;
                const dy = b.y - a.y;
                const minDist = (a.radius || radius) + (b.radius || radius);
                const distSq = dx * dx + dy * dy;
                if (distSq > 0 && distSq < minDist * minDist) {
                    const dist = Math.sqrt(distSq);
                    const overlap = minDist - dist;
                    const nx = dx / dist;
                    const ny = dy / dist;
                    a.x -= nx * overlap * 0.5;
                    a.y -= ny * overlap * 0.5;
                    b.x += nx * overlap * 0.5;
                    b.y += ny * overlap * 0.5;
                    a.vx -= nx * push;
                    a.vy -= ny * push;
                    b.vx += nx * push;
                    b.vy += ny * push;
                }
            }
        }
    }

    function countAdjacentPrey(agent) {
        let count = 0;
        const radiusSq = ADJACENT_RADIUS * ADJACENT_RADIUS;
        for (let other of agents) {
            if (other === agent || !other.alive) continue;
            const dx = other.x - agent.x;
            const dy = other.y - agent.y;
            if (dx * dx + dy * dy <= radiusSq) {
                count++;
            }
        }
        return count;
    }

    function isNearEdge(x, y, radius) {
        const margin = Math.max(0, EDGE_EAT_MARGIN);
        return (
            x <= simBounds.x + radius + margin ||
            x >= simBounds.x + simBounds.width - radius - margin ||
            y <= simBounds.y + radius + margin ||
            y >= simBounds.y + simBounds.height - radius - margin
        );
    }

    function drawTriangle(x, y, angle, size, color) {
        const tip = size * 1.2;
        const base = size * 0.9;
        const halfBase = base * 0.6;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(tip, 0);
        ctx.lineTo(-base, -halfBase);
        ctx.lineTo(-base, halfBase);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
        ctx.restore();
    }

    function loop(timestamp) {
        if (!running) {
            rafId = null;
            return;
        }

        const width = viewport.width;
        const height = viewport.height;
        const deltaSec = Number.isFinite(timestamp) && Number.isFinite(lastTimestamp)
            ? Math.min((timestamp - lastTimestamp) / 1000, 0.05)
            : (config?.simulation?.dt ?? 0.016);
        lastTimestamp = Number.isFinite(timestamp) ? timestamp : lastTimestamp;

        if (!userEnabled) {
            ctx.clearRect(0, 0, width, height);
            rafId = requestAnimationFrame(loop);
            return;
        }

        ctx.clearRect(0, 0, width, height);

        const cornerRadius = getCornerRadius();
        ctx.save();
        buildRoundedRectPath(cornerRadius);
        ctx.clip();

        ctx.save();
        ctx.fillStyle = '#ffffff';
        ctx.font = '14px "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const aliveCount = agents.reduce((acc, a) => acc + (a.alive ? 1 : 0), 0);
        ctx.fillText(`Prey: ${aliveCount}`, simBounds.x + 14, simBounds.y + 12);
        ctx.fillText(`Predators: ${predators.length}`, simBounds.x + 14, simBounds.y + 30);
        ctx.restore();

        if (config?.debug === true) {
            const aliveAgents = agents.filter(a => a.alive);
            if (aliveAgents.length) {
                let candidates = [];
                for (const agent of aliveAgents) {
                    const { enemyBins, friendlyBins } = getVisionBins(agent);
                    const hasSignal = enemyBins.some(v => v > 0) || friendlyBins.some(v => v > 0);
                    if (hasSignal) {
                        candidates.push({ agent, enemyBins, friendlyBins });
                    }
                }
                const selectionPool = candidates.length ? candidates : aliveAgents.map(agent => {
                    const { enemyBins, friendlyBins } = getVisionBins(agent);
                    return { agent, enemyBins, friendlyBins };
                });
                const chosen = selectionPool[Math.floor(Math.random() * selectionPool.length)];
                const debugAgent = chosen.agent;
                const enemyBins = chosen.enemyBins;
                const friendlyBins = chosen.friendlyBins;
                const twoPi = Math.PI * 2;
                ctx.save();
                ctx.lineWidth = 0.5;
                ctx.globalAlpha = 0.5;
                for (let i = 0; i < VISION_PIXELS; i++) {
                    const angle = ((i + 0.5) / VISION_PIXELS) * twoPi;
                    const cos = Math.cos(angle);
                    const sin = Math.sin(angle);
                    const enemyDist = enemyBins[i];
                    const friendlyDist = friendlyBins[i];
                    if (enemyDist > 0) {
                        const len = enemyDist * VISION_RADIUS;
                        ctx.strokeStyle = 'rgba(255, 90, 70, 0.6)';
                        ctx.beginPath();
                        ctx.moveTo(debugAgent.x, debugAgent.y);
                        ctx.lineTo(debugAgent.x + cos * len, debugAgent.y + sin * len);
                        ctx.stroke();
                    }
                    if (friendlyDist > 0) {
                        const len = friendlyDist * VISION_RADIUS;
                        ctx.strokeStyle = 'rgba(120, 200, 255, 0.6)';
                        ctx.beginPath();
                        ctx.moveTo(debugAgent.x, debugAgent.y);
                        ctx.lineTo(debugAgent.x + cos * len, debugAgent.y + sin * len);
                        ctx.stroke();
                    }
                }
                ctx.restore();
            }
        }

        predators.forEach(p => {
            let bestTarget = null;
            let minDist = Infinity;

            if (p.stun > 0) {
                p.stun = Math.max(0, p.stun - deltaSec);
                p.vx *= 0.9;
                p.vy *= 0.9;
            } else {
                agents.forEach(a => {
                    if (!a.alive) return;
                    const dx = a.x - p.x;
                    const dy = a.y - p.y;
                    const dist = dx * dx + dy * dy;
                    if (dist < minDist) {
                        minDist = dist;
                        bestTarget = a;
                    }
                });

                if (bestTarget) {
                    let dx = bestTarget.x - p.x;
                    let dy = bestTarget.y - p.y;
                    let mag = Math.sqrt(dx * dx + dy * dy) + 1e-5;
                    p.vx += (dx / mag) * 0.25;
                    p.vy += (dy / mag) * 0.25;
                }
            }

            let s = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
            if (s > config.agents.predator_speed) {
                p.vx = (p.vx / s) * config.agents.predator_speed;
                p.vy = (p.vy / s) * config.agents.predator_speed;
            }
            p.x += p.vx;
            p.y += p.vy;
            applyWallRepulsion(p, 0.2);

            const predatorAngle = Math.atan2(p.vy, p.vx) || 0;
            const predatorColor = p.stun > 0 ? '#b36b40' : '#ff6600';
            drawTriangle(p.x, p.y, predatorAngle, p.radius || PREDATOR_RADIUS, predatorColor);
        });

        resolveCollisions(predators, PREDATOR_RADIUS, 0.3);
        predators.forEach(p => applyWallRepulsion(p, 0));

        agents.forEach(a => {
            if (!a.alive) return;

            let inputs = getSensors(a);
            let outputs = brain.forward(inputs);

            a.vx += outputs[0] * 0.5 * FISH_SPEED;
            a.vy += outputs[1] * 0.5 * FISH_SPEED;

            applyWallRepulsion(a, 0.2);

            a.vx *= 0.95;
            a.vy *= 0.95;

            a.x += a.vx;
            a.y += a.vy;

            const neighbors = countAdjacentPrey(a);
            const speed = Math.sqrt(a.vx * a.vx + a.vy * a.vy);
            const densityFactor = Math.min(neighbors / ADJACENT_MAX, 1);
            const speedFactor = Math.min(speed / SPEED_SCALE, 1);
            const escapeChance = Math.min(0.9, densityFactor * DENSITY_WEIGHT + speedFactor * SPEED_WEIGHT);
            const eatenProbBase = Math.min(1, BASE_EAT_PROB * (1 - escapeChance));
            const eatenProbFinal = isNearEdge(a.x, a.y, a.radius || FISH_RADIUS) ? 1 : eatenProbBase;

            predators.forEach(p => {
                const dx = a.x - p.x;
                const dy = a.y - p.y;
                const d = Math.sqrt(dx * dx + dy * dy);
                const eatDistance = (a.radius || FISH_RADIUS) + (p.radius || PREDATOR_RADIUS);
                if (d < eatDistance) {
                    if ((p.stun || 0) > 0) {
                        return;
                    }
                    if (Math.random() < eatenProbFinal) {
                        a.alive = false;
                    } else {
                        p.stun = Math.max(p.stun || 0, PREDATOR_STUN_SECONDS);
                        if (d > 0) {
                        a.vx += (dx / d) * 1.2;
                        a.vy += (dy / d) * 1.2;
                        }
                    }
                }
            });

            const boldness = Math.min(1, Math.max(0, 1 - eatenProbFinal));
            let r = Math.floor(70 + 185 * boldness);
            let g = Math.floor(80 + 120 * boldness);
            let b = Math.floor(90 + 165 * boldness);

            const preyAngle = Math.atan2(a.vy, a.vx) || 0;
            drawTriangle(a.x, a.y, preyAngle, a.radius || FISH_RADIUS, `rgb(${r}, ${g}, ${b})`);
        });

        resolveCollisions(agents, FISH_RADIUS, 0.2);
        agents.forEach(a => applyWallRepulsion(a, 0));

        ctx.restore();

        ctx.save();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        buildRoundedRectPath(cornerRadius);
        ctx.stroke();
        ctx.restore();

        rafId = requestAnimationFrame(loop);
    }

    function updateRunningState() {
        const shouldRun = !document.hidden;
        if (shouldRun && !running) {
            running = true;
            rafId = requestAnimationFrame(loop);
        } else if (!shouldRun && running) {
            running = false;
        }
        if (toggleButton) {
            toggleButton.setAttribute('aria-pressed', String(userEnabled));
        }
        if (profileImage) {
            profileImage.classList.toggle('is-dim', userEnabled);
        }
    }

    async function init() {
        resizeCanvas();
        config = await fetchJsonWithFallback(CONFIG_URL, DEFAULT_CONFIG);
        setAspectRatioFromConfig();
        updateSimBounds();
        if (config && config.agents) {
            const minDim = Math.max(1, Math.min(simBounds.width || viewport.width, simBounds.height || viewport.height));
            if (Number.isFinite(config.agents.fish_radius)) {
                FISH_RADIUS_RATIO = config.agents.fish_radius <= 1 ? config.agents.fish_radius : config.agents.fish_radius / minDim;
            }
            if (Number.isFinite(config.agents.predator_radius)) {
                PREDATOR_RADIUS_RATIO = config.agents.predator_radius <= 1 ? config.agents.predator_radius : config.agents.predator_radius / minDim;
            }
            if (Number.isFinite(config.agents.predator_stun_seconds)) {
                PREDATOR_STUN_SECONDS = config.agents.predator_stun_seconds;
            }
            if (Number.isFinite(config.agents.edge_eat_margin)) {
                EDGE_EAT_MARGIN = config.agents.edge_eat_margin;
            }
            if (Number.isFinite(config.agents.fish_speed)) {
                FISH_SPEED = config.agents.fish_speed;
            }
            if (Number.isFinite(config.agents.corner_radius_factor)) {
                CORNER_RADIUS_FACTOR = config.agents.corner_radius_factor;
            }
            if (Number.isFinite(config.agents.corner_radius_min)) {
                CORNER_RADIUS_MIN = config.agents.corner_radius_min;
            }
            if (Number.isFinite(config.agents.corner_radius_max)) {
                CORNER_RADIUS_MAX = config.agents.corner_radius_max;
            }
            if (Number.isFinite(config.agents.adjacent_radius)) {
                ADJACENT_RADIUS_RATIO = config.agents.adjacent_radius <= 1 ? config.agents.adjacent_radius : config.agents.adjacent_radius / minDim;
            } else {
                ADJACENT_RADIUS_RATIO = (FISH_RADIUS_RATIO || 0.013) * 4;
            }
            if (Number.isFinite(config.agents.vision_radius)) {
                VISION_RADIUS = config.agents.vision_radius <= 1
                    ? config.agents.vision_radius * minDim
                    : config.agents.vision_radius;
            }
            if (Number.isFinite(config.agents.vision_pixels)) {
                VISION_PIXELS = Math.max(4, Math.round(config.agents.vision_pixels));
            }
            if (Number.isFinite(config.agents.adjacent_max)) {
                ADJACENT_MAX = config.agents.adjacent_max;
            }
            if (Number.isFinite(config.agents.speed_scale)) {
                SPEED_SCALE = config.agents.speed_scale;
            } else if (Number.isFinite(config.agents.fish_speed)) {
                SPEED_SCALE = config.agents.fish_speed * 1.5;
            }
            if (Number.isFinite(config.agents.base_eat_prob)) {
                BASE_EAT_PROB = config.agents.base_eat_prob;
            }
            if (Number.isFinite(config.agents.density_weight)) {
                DENSITY_WEIGHT = config.agents.density_weight;
            }
            if (Number.isFinite(config.agents.speed_weight)) {
                SPEED_WEIGHT = config.agents.speed_weight;
            }
        }
        updateScaledLengths();
        const weights = await fetchJsonWithFallback(WEIGHTS_URL, null);
        if (!weights) {
            setStatus('Missing weights.json');
            return;
        }

        brain = new NeuralNet(weights);
        resetSimulation();
        setStatus('Running Live Inference');
        updateRunningState();
    }

    window.resetSimulation = resetSimulation;

    window.addEventListener('resize', () => {
        resizeCanvas();
        updateSimBounds();
        updateScaledLengths();
        if (agents.length || predators.length) {
            agents.forEach(a => applyWallRepulsion(a, 0));
            predators.forEach(p => applyWallRepulsion(p, 0));
        }
    });

    document.addEventListener('visibilitychange', updateRunningState);

    if (toggleButton) {
        toggleButton.addEventListener('click', () => {
            const wasEnabled = userEnabled;
            userEnabled = !userEnabled;
            if (!wasEnabled && userEnabled) {
                resetSimulation();
            }
            updateRunningState();
        });
    }

    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        predators.push({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
            vx: 0,
            vy: 0,
            radius: PREDATOR_RADIUS,
            stun: 0
        });
        applyWallRepulsion(predators[predators.length - 1], 0);
    });

    init();
})();
