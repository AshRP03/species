"""Reproduction and structural mutation.

Evolution operates on topology (add/remove nodes and edges), on a slower
timescale than the within-lifetime Hebbian weight updates. When an organism
banks enough surplus energy it splits: half the energy goes to a mutated copy.
There is no fitness function — only the fact that survivors are the ones that
managed to replicate.
"""

from __future__ import annotations

import copy

import numpy as np

import config
from organism import Neuron, Organism, Synapse


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def nearby_position(pos: tuple[int, int], rng: np.random.Generator, grid_size: int) -> tuple[int, int]:
    if config.DISPERSE_RANDOM:
        # Scatter offspring onto open ground: they must forage to survive.
        return (int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size)))
    r = config.SPAWN_RADIUS
    x = _clamp(pos[0] + int(rng.integers(-r, r + 1)), 0, grid_size - 1)
    y = _clamp(pos[1] + int(rng.integers(-r, r + 1)), 0, grid_size - 1)
    return (x, y)


def maybe_replicate(
    org: Organism,
    rng: np.random.Generator,
    new_id: int,
    grid_size: int,
) -> Organism | None:
    if org.energy < config.REPLICATION_THRESHOLD:
        return None

    org.energy /= 2.0
    child = copy.deepcopy(org)
    child.id = new_id
    child.generation = org.generation + 1
    child.age = 0
    child.energy = org.energy
    child.position = nearby_position(org.position, rng, grid_size)
    # Learning state is within-lifetime: start the child fresh.
    child.reward_baseline = 0.0
    child.steps_since_fed = 0
    for s in child.synapses:
        s.elig = 0.0
    mutate(child, rng)
    child.refresh_role_cache()
    return child


def mutate(org: Organism, rng: np.random.Generator) -> None:
    """Two mutation regimes, deliberately separated:

    1. CONTINUOUS micro-mutation (always, tiny): every heritable continuous
       parameter -- synapse weights and per-neuron threshold / memory_decay /
       noise_scale -- drifts by a small gaussian step on EVERY birth. Any one
       generation barely changes, but selection accumulates the beneficial
       drift across many generations: a smooth evolutionary walk.
    2. RARE structural (macro) mutation: a discrete topology change (add/remove
       edge, add node) at a low per-birth rate, so innovations are occasional
       rather than a fresh upheaval every generation.

    This replaces the old scheme where a single 10%-per-birth dice-roll picked
    one discrete change (often a large topological jump) -- "a new uncertainty
    every generation" -- with subtle change that stacks over time plus infrequent
    structural leaps.
    """
    _micro_mutate(org, rng)

    if rng.random() < config.STRUCTURAL_MUTATION_RATE:
        choice = rng.integers(0, 3)
        if choice == 0:
            _add_edge(org, rng)
        elif choice == 1:
            _remove_weakest_edge(org)
        else:
            _add_node(org, rng)


def _valid_dst(org: Organism, nid: int) -> bool:
    return org.neurons[nid].neuron_type != "sensor"


def _valid_src(org: Organism, nid: int) -> bool:
    return org.neurons[nid].neuron_type != "actuator"


def _add_edge(org: Organism, rng: np.random.Generator) -> None:
    ids = list(org.neurons.keys())
    if len(ids) < 2:
        return
    existing = {(s.src, s.dst) for s in org.synapses}
    for _ in range(8):  # a few tries to find a legal, non-duplicate edge
        src = int(rng.choice(ids))
        dst = int(rng.choice(ids))
        if src == dst or not _valid_src(org, src) or not _valid_dst(org, dst):
            continue
        if (src, dst) in existing:
            continue
        w = float(rng.normal(0.0, config.INIT_WEIGHT_SCALE))
        org.synapses.append(Synapse(src=src, dst=dst, weight=w))
        return


def _remove_weakest_edge(org: Organism) -> None:
    if not org.synapses:
        return
    weakest = min(range(len(org.synapses)), key=lambda i: abs(org.synapses[i].weight))
    org.synapses.pop(weakest)


def _add_node(org: Organism, rng: np.random.Generator) -> None:
    """Insert an interneuron and splice it onto the graph via one in/out edge.

    Splitting an existing edge (src -> new -> dst) keeps the new node functional
    rather than dangling, which makes added structure more likely to survive.
    """
    nid = org.next_neuron_id
    org.next_neuron_id += 1
    org.neurons[nid] = Neuron(
        id=nid,
        neuron_type="inter",
        threshold=float(rng.uniform(0.2, 0.8)),
        bias=float(rng.normal(0.0, 0.1)),
        memory_decay=float(rng.uniform(*config.INTERNEURON_MEMORY_DECAY_RANGE)),
    )

    if org.synapses:
        old = org.synapses[int(rng.integers(0, len(org.synapses)))]
        org.synapses.append(Synapse(src=old.src, dst=nid, weight=old.weight))
        org.synapses.append(Synapse(src=nid, dst=old.dst, weight=float(rng.normal(0.0, config.INIT_WEIGHT_SCALE))))
    else:
        # No edges to splice; wire a random sensor->new and new->random actuator.
        if org.sensor_ids:
            org.synapses.append(
                Synapse(src=int(rng.choice(org.sensor_ids)), dst=nid,
                        weight=float(rng.normal(0.0, config.INIT_WEIGHT_SCALE)))
            )
        if org.actuator_ids:
            org.synapses.append(
                Synapse(src=nid, dst=int(rng.choice(org.actuator_ids)),
                        weight=float(rng.normal(0.0, config.INIT_WEIGHT_SCALE)))
            )


def _micro_mutate(org: Organism, rng: np.random.Generator) -> None:
    """Small gaussian drift on EVERY heritable continuous parameter, applied on
    every birth. Individually tiny; accumulates across generations under
    selection. Covers synapse weights and each non-sensor neuron's intrinsic
    properties (threshold, plus memory_decay for interneurons / noise_scale for
    actuators -- the genome traits that used to be frozen at birth)."""
    s = config.MICRO_MUTATION_SCALE
    if s <= 0.0:
        return
    clip = config.WEIGHT_CLIP

    if org.synapses:
        noise = rng.normal(0.0, s, size=len(org.synapses))
        for syn, dw in zip(org.synapses, noise):
            syn.weight = float(np.clip(syn.weight + dw, -clip, clip))

    for n in org.neurons.values():
        if n.neuron_type == "sensor":
            continue
        n.threshold = float(np.clip(n.threshold + rng.normal(0.0, s), 0.02, 1.5))
        if n.neuron_type == "inter":
            n.memory_decay = float(np.clip(n.memory_decay + rng.normal(0.0, s), 0.0, 0.98))
        elif n.neuron_type == "actuator":
            n.noise_scale = float(np.clip(n.noise_scale + rng.normal(0.0, s), 0.0, 1.0))
