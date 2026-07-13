"""Perception-action instrumentation.

The point of ADNS is to watch for sensor input causally driving adaptive
behaviour, and eventually for distinct sub-regions of the graph to specialise.
This module measures that at three levels:

  1. BEHAVIOURAL  — does the organism actually move toward sensed energy?
     `sensorimotor_alignment` compares the sensor gradient vector with the
     realised movement vector. This is the headline "is there a loop" signal:
     a blind random-walker scores ~0, a perceiving forager scores > 0.

  2. STRUCTURAL   — is there wiring that *could* carry a loop?
     `sensor_actuator_reach` counts sensor->actuator directed paths. This is the
     substrate; it can exist without being used.

  3. CAUSAL       — does one sensor actually drive an actuator right now?
     `influence_matrix` probes the network by activating one sensor at a time
     and reading the actuator response. This is the first step toward
     attributing a function ("turn north when energy is north") to structure,
     and later toward finding functional modules.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from environment import _DIRS  # N, S, E, W unit vectors, screen convention
from organism import Organism, propagate


# ---------------------------------------------------------------------------
# 1. Behavioural: sensor gradient vs. realised movement
# ---------------------------------------------------------------------------
def sensor_vector(sensor_readings) -> np.ndarray:
    """Weighted sum of the 4 directional sensor readings -> a 2D "where energy is"
    vector in world coordinates."""
    r = np.asarray(sensor_readings, dtype=np.float64)[:4]
    if r.shape[0] < 4:
        r = np.pad(r, (0, 4 - r.shape[0]))
    return r @ _DIRS  # (2,)


def alignment(sensor_vec: np.ndarray, move_vec: np.ndarray) -> float | None:
    """Cosine similarity between where energy was sensed and where the organism
    moved. Returns None when either vector is zero (no signal / no motion)."""
    sn = np.linalg.norm(sensor_vec)
    mn = np.linalg.norm(move_vec)
    if sn < 1e-9 or mn < 1e-9:
        return None
    return float(np.dot(sensor_vec, move_vec) / (sn * mn))


# ---------------------------------------------------------------------------
# 2. Structural: sensor -> actuator reachability
# ---------------------------------------------------------------------------
def _adjacency(org: Organism) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {nid: [] for nid in org.neurons}
    for s in org.synapses:
        adj[s.src].append(s.dst)
    return adj


def sensor_actuator_reach(org: Organism) -> tuple[bool, int]:
    """Return (has_any_path, number_of_reachable_sensor->actuator pairs)."""
    adj = _adjacency(org)
    actuators = set(org.actuator_ids)
    pairs = 0
    for s in org.sensor_ids:
        seen = {s}
        dq = deque([s])
        reached = set()
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v in actuators:
                    reached.add(v)
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        pairs += len(reached)
    return pairs > 0, pairs


# ---------------------------------------------------------------------------
# 3. Causal: single-sensor perturbation -> actuator response
# ---------------------------------------------------------------------------
def influence_matrix(org: Organism, passes: int = 3) -> np.ndarray:
    """(n_sensor, n_actuator) matrix: activate one sensor, run a few propagation
    passes (so signal can traverse depth), read which actuators fire.

    Non-zero entries mean a sensor causally reaches an actuator through the
    current weights and thresholds — a functioning reflex arc. Reading the
    diagonal-like structure (does the N sensor drive the N actuator?) tells you
    whether the reflex is *adaptive* or just wired.
    """
    sensors = org.sensor_ids
    actuators = org.actuator_ids
    mat = np.zeros((len(sensors), len(actuators)))
    saved = {nid: n.state for nid, n in org.neurons.items()}
    try:
        for i, sid in enumerate(sensors):
            for nid, n in org.neurons.items():
                n.state = 0.0
            org.neurons[sid].state = 1.0
            for _ in range(passes):
                # keep the probed sensor clamped high across passes
                propagate(org)
                org.neurons[sid].state = 1.0
            for j, aid in enumerate(actuators):
                mat[i, j] = org.neurons[aid].state
    finally:
        for nid, n in org.neurons.items():
            n.state = saved[nid]
    return mat


# ---------------------------------------------------------------------------
# Population-level summaries
# ---------------------------------------------------------------------------
def structural_summary(population) -> dict:
    if not population:
        return {"frac_with_loop": 0.0, "mean_sa_pairs": 0.0}
    flags = []
    pairs = []
    for o in population:
        has, n = sensor_actuator_reach(o)
        flags.append(1.0 if has else 0.0)
        pairs.append(n)
    return {
        "frac_with_loop": float(np.mean(flags)),
        "mean_sa_pairs": float(np.mean(pairs)),
    }


def causal_summary(population, sample: int = 40, rng: np.random.Generator | None = None) -> dict:
    """Mean causal sensor->actuator influence density, and how often the probe
    shows a *direction-matched* reflex (sensor_k drives actuator_k), sampled over
    the population."""
    if not population:
        return {"mean_influence": 0.0, "matched_reflex_rate": 0.0}
    pop = population
    if rng is not None and len(pop) > sample:
        idx = rng.choice(len(pop), size=sample, replace=False)
        pop = [population[i] for i in idx]
    dens = []
    matched = []
    for o in pop:
        m = influence_matrix(o)
        if m.size == 0:
            continue
        dens.append(float(m.mean()))
        k = min(m.shape[0], m.shape[1])
        # direction-matched = sensor i drives the same-index actuator (N->N ...)
        matched.append(float(np.mean([m[i, i] for i in range(k)])) if k else 0.0)
    return {
        "mean_influence": float(np.mean(dens)) if dens else 0.0,
        "matched_reflex_rate": float(np.mean(matched)) if matched else 0.0,
    }
