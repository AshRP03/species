"""Organism = a sparse directed graph of neurons living in the 2D world.

The graph has three neuron roles:
  - sensor:   input node, state is set externally from the environment
  - inter:    hidden node, updated by threshold activation
  - actuator: output node, its state drives movement

Signal propagation is a single synchronous threshold pass per timestep. Learning
is a purely local Hebbian rule gated by whether the organism gained or lost
energy this step (a stand-in for a diffuse neuromodulator like dopamine).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class Neuron:
    id: int
    threshold: float = 0.5
    bias: float = 0.0
    state: float = 0.0            # current activation (0.0 or 1.0 after a pass)
    neuron_type: str = "inter"    # 'sensor' | 'inter' | 'actuator'


@dataclass
class Synapse:
    src: int
    dst: int
    weight: float


@dataclass
class Organism:
    id: int
    neurons: dict[int, Neuron]
    synapses: list[Synapse]
    energy: float
    position: tuple[int, int]
    age: int = 0
    generation: int = 0
    next_neuron_id: int = 0       # monotonic counter for adding neurons via mutation
    # cached role -> neuron ids, kept in sync as structure changes
    sensor_ids: list[int] = field(default_factory=list)
    actuator_ids: list[int] = field(default_factory=list)

    def refresh_role_cache(self) -> None:
        self.sensor_ids = [n.id for n in self.neurons.values() if n.neuron_type == "sensor"]
        self.actuator_ids = [n.id for n in self.neurons.values() if n.neuron_type == "actuator"]


def make_random_organism(org_id: int, rng: np.random.Generator, grid_size: int) -> Organism:
    """Build a founder organism: SENSOR_COUNT sensors, ACTUATOR_COUNT actuators,
    the rest interneurons, wired with a sparse random directed graph."""
    n_total = config.INIT_NODES
    n_sensor = config.SENSOR_COUNT
    n_actuator = config.ACTUATOR_COUNT
    n_inter = max(0, n_total - n_sensor - n_actuator)

    neurons: dict[int, Neuron] = {}
    nid = 0

    # Sensors occupy the first ids so environment code can index them by order.
    for _ in range(n_sensor):
        neurons[nid] = Neuron(id=nid, neuron_type="sensor", threshold=0.0)
        nid += 1
    for _ in range(n_actuator):
        neurons[nid] = Neuron(
            id=nid,
            neuron_type="actuator",
            threshold=float(rng.uniform(0.2, 0.8)),
            bias=float(rng.normal(0.0, 0.1)),
        )
        nid += 1
    for _ in range(n_inter):
        neurons[nid] = Neuron(
            id=nid,
            neuron_type="inter",
            threshold=float(rng.uniform(0.2, 0.8)),
            bias=float(rng.normal(0.0, 0.1)),
        )
        nid += 1

    # Sparse random edges. Sensors only send; actuators only receive.
    synapses: list[Synapse] = []
    ids = list(neurons.keys())
    for src in ids:
        if neurons[src].neuron_type == "actuator":
            continue
        for dst in ids:
            if src == dst:
                continue
            if neurons[dst].neuron_type == "sensor":
                continue
            if rng.random() < config.INIT_EDGE_PROB:
                w = float(rng.normal(0.0, config.INIT_WEIGHT_SCALE))
                synapses.append(Synapse(src=src, dst=dst, weight=w))

    pos = (int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size)))
    org = Organism(
        id=org_id,
        neurons=neurons,
        synapses=synapses,
        energy=config.INIT_ENERGY,
        position=pos,
        next_neuron_id=nid,
    )
    org.refresh_role_cache()
    return org


def propagate(org: Organism) -> None:
    """One synchronous threshold pass. Sensor states must already be set."""
    inputs: dict[int, float] = defaultdict(float)
    neurons = org.neurons
    for syn in org.synapses:
        s = neurons[syn.src].state
        if s > 0.0:
            inputs[syn.dst] += s * syn.weight

    for nid, neuron in neurons.items():
        if neuron.neuron_type == "sensor":
            continue  # driven externally
        total = inputs[nid] + neuron.bias
        neuron.state = 1.0 if total >= neuron.threshold else 0.0


def update_weights(org: Organism, delta_energy: float) -> None:
    """Local Hebbian update gated by the sign of the energy change this step.

    Pre & post both fired while energy rose -> strengthen. While energy fell ->
    weaken. Synapses that decay below the prune threshold are removed, which is
    the mechanism that lets sparsity emerge over a lifetime.
    """
    gate = 1.0 if delta_energy >= 0 else -1.0
    lr = config.LEARNING_RATE
    clip = config.WEIGHT_CLIP
    prune = config.WEIGHT_PRUNE_THRESHOLD
    neurons = org.neurons

    survivors: list[Synapse] = []
    for syn in org.synapses:
        pre = neurons[syn.src].state
        post = neurons[syn.dst].state
        if pre > 0.0 and post > 0.0:
            syn.weight += lr * gate
            if syn.weight > clip:
                syn.weight = clip
            elif syn.weight < -clip:
                syn.weight = -clip
        if abs(syn.weight) >= prune:
            survivors.append(syn)
    org.synapses = survivors
