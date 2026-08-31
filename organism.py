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
    # Working memory (interneurons only): state = memory_decay*old_state +
    # (1-memory_decay)*new_activation, instead of being fully overwritten each
    # step. At 0.0 this is identical to plain reactive threshold behaviour.
    # Sensors and actuators keep memory_decay=0.0 (pure instantaneous input /
    # fast readout respectively); only interneurons get a nonzero value.
    memory_decay: float = 0.0
    # Actuators only: std of the fresh per-timestep exploration noise added to
    # this neuron's input (see propagate()). Heritable and mutable -- an
    # organism can evolve from noisy babbling toward confident, near-
    # deterministic control as its wiring becomes reliable, or vice versa.
    noise_scale: float = 0.0


@dataclass
class Synapse:
    src: int
    dst: int
    weight: float
    elig: float = 0.0            # eligibility trace: decaying memory of pre*post


@dataclass
class Organism:
    id: int
    neurons: dict[int, Neuron]
    synapses: list[Synapse]
    energy: float
    position: tuple[int, int]
    age: int = 0
    generation: int = 0
    reward_baseline: float = 0.0  # EMA of recent energy delta (the neuromodulator baseline)
    steps_since_fed: int = 0      # hunger clock; resets on any successful harvest, else counts up
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
        # Actuators are pure readouts, symmetric to how sensors are pure inputs:
        # no fixed bias of their own (a fixed bias, drawn once at birth, can only
        # ever settle into "always fires" or "never fires" -- it can't produce
        # ongoing exploration). Their activation is a weighted sum of interneuron
        # /sensor input PLUS fresh per-timestep noise (see propagate()), which is
        # what actually sustains "motor babbling" instead of it fizzling out.
        # Threshold and noise_scale are independently randomized per actuator --
        # both are heritable/mutable genome traits now (see
        # evolution._micro_mutate), so any directional bias between
        # opposing actuators is something selection can equalize OR exploit,
        # not a permanent birth-lottery accident (that used to require forcing
        # all four to share one value; not needed anymore since mutation can
        # now correct a bad roll instead of being stuck with it for life).
        neurons[nid] = Neuron(
            id=nid,
            neuron_type="actuator",
            threshold=float(rng.uniform(*config.ACTUATOR_THRESHOLD_RANGE)),
            bias=0.0,
            noise_scale=float(rng.uniform(*config.ACTUATOR_NOISE_SCALE_RANGE)),
        )
        nid += 1
    for _ in range(n_inter):
        neurons[nid] = Neuron(
            id=nid,
            neuron_type="inter",
            threshold=float(rng.uniform(0.2, 0.8)),
            bias=float(rng.normal(0.0, 0.1)),
            # Working memory: independently randomized per neuron, so a lineage
            # gets a MIX of fast-reactive (low decay) and slow-integrating (high
            # decay) interneurons for evolution/learning to make use of.
            memory_decay=float(rng.uniform(*config.INTERNEURON_MEMORY_DECAY_RANGE)),
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

    # Guarantee every actuator has at least one incoming edge. With only 5%
    # random edge probability and 4 actuators, roughly 1-in-6 would otherwise
    # end up completely disconnected -- permanently unreachable by any signal,
    # no matter how wiring or weights evolve later.
    actuator_ids_here = [nid for nid, n in neurons.items() if n.neuron_type == "actuator"]
    eligible_sources = [nid for nid, n in neurons.items() if n.neuron_type != "actuator"]
    for aid in actuator_ids_here:
        if not any(s.dst == aid for s in synapses):
            src = int(rng.choice(eligible_sources))
            w = float(rng.normal(0.0, config.INIT_WEIGHT_SCALE))
            synapses.append(Synapse(src=src, dst=aid, weight=w))

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


def propagate(org: Organism, rng: np.random.Generator) -> None:
    """One synchronous threshold pass. Sensor states must already be set.

    Actuators get a FRESH random nudge every call (not a fixed per-neuron bias),
    which is what sustains ongoing "motor babbling" instead of it settling into
    a permanent always-on or always-off state. A fixed bias can only ever
    determine one outcome once at birth; noise redrawn every timestep keeps
    exploration alive so Hebbian learning has continuous co-activation events
    to shape into purposeful movement.
    """
    inputs: dict[int, float] = defaultdict(float)
    neurons = org.neurons
    for syn in org.synapses:
        s = neurons[syn.src].state
        if s > 0.0:
            inputs[syn.dst] += s * syn.weight

    for nid, neuron in neurons.items():
        if neuron.neuron_type == "sensor":
            continue  # driven externally
        if neuron.neuron_type == "actuator":
            nudge = rng.normal(0.0, neuron.noise_scale)
        else:
            nudge = neuron.bias
        total = inputs[nid] + nudge
        activation = 1.0 if total >= neuron.threshold else 0.0
        if neuron.neuron_type == "inter" and neuron.memory_decay > 0.0:
            # Working memory: blend with the previous state instead of a hard
            # overwrite, so information can persist across timesteps rather
            # than being fully forgotten every single pass.
            neuron.state = neuron.memory_decay * neuron.state + (1.0 - neuron.memory_decay) * activation
        else:
            neuron.state = activation


def update_weights(org: Organism, delta_energy: float) -> None:
    """Local, reward-modulated Hebbian update with eligibility traces.

    Each synapse carries an eligibility trace -- a decaying memory of recent
    pre*post coincidence. The neuromodulator is the *advantage* of this step's
    energy change over the organism's own running baseline (a prediction error),
    so behaviour that improves the organism's fortune is reinforced even before
    net energy turns positive. This gives the temporal credit assignment a pure
    per-step gate cannot: the approach that led to food gets credited, not just
    the instant of arrival. Weights below the prune threshold are removed.
    """
    lr = config.LEARNING_RATE
    decay = config.ELIGIBILITY_DECAY
    clip = config.WEIGHT_CLIP
    prune = config.WEIGHT_PRUNE_THRESHOLD
    neurons = org.neurons

    # Diffuse neuromodulator: how much better (or worse) than recently expected.
    org.reward_baseline += config.REWARD_BASELINE_RATE * (delta_energy - org.reward_baseline)
    modulator = delta_energy - org.reward_baseline

    survivors: list[Synapse] = []
    for syn in org.synapses:
        pre = neurons[syn.src].state
        post = neurons[syn.dst].state
        syn.elig = decay * syn.elig + pre * post
        syn.weight += lr * modulator * syn.elig
        if syn.weight > clip:
            syn.weight = clip
        elif syn.weight < -clip:
            syn.weight = -clip
        if abs(syn.weight) >= prune:
            survivors.append(syn)
    org.synapses = survivors
