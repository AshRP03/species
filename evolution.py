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
    mutate(child, rng)
    child.refresh_role_cache()
    return child


def mutate(org: Organism, rng: np.random.Generator) -> None:
    """With probability MUTATION_RATE, apply one structural/parametric change."""
    if rng.random() >= config.MUTATION_RATE:
        return

    choice = rng.integers(0, 4)
    if choice == 0:
        _add_edge(org, rng)
    elif choice == 1:
        _remove_weakest_edge(org)
    elif choice == 2:
        _add_node(org, rng)
    else:
        _perturb_weights(org, rng)


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


def _perturb_weights(org: Organism, rng: np.random.Generator) -> None:
    if not org.synapses:
        return
    noise = rng.normal(0.0, config.WEIGHT_PERTURB_SCALE, size=len(org.synapses))
    clip = config.WEIGHT_CLIP
    for syn, dw in zip(org.synapses, noise):
        syn.weight = float(np.clip(syn.weight + dw, -clip, clip))
