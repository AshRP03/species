# SPECIES

**Sparse Pattern-Emergent Cortical Intelligence via Evolutionary Selection**

SPECIES is an experimental simulation of adaptive agents in a resource-constrained 2D environment. Instead of training a fixed neural-network architecture with backpropagation, the system evolves sparse neural graphs through energy economics, structural mutation, and local Hebbian learning.

> **Research question:** Can perception-action loops emerge when agents must survive, move, and reproduce in an environment?

## Why this project exists

The goal is not simply to produce sparse networks. SPECIES is an instrumented environment for studying whether sensor inputs can become causally connected to adaptive behavior—and whether specialized functional regions can emerge within an evolving graph.

The project currently treats the following as separate signals:

- **Behavior:** Does movement align with sensed resources?
- **Structure:** Does a directed sensor-to-actuator path exist?
- **Causality:** Does activating a sensor influence the corresponding actuator?

This separation makes it possible to distinguish a network that contains the *substrate* for intelligence from one that is actually using it.

## Current model

Each organism:

1. Exists as a sparse directed neural graph.
2. Receives directional sensor input from a 2D environment.
3. Propagates activity through threshold-based neurons.
4. Moves and harvests energy from environmental patches.
5. Updates connections through local Hebbian learning.
6. Replicates when it accumulates sufficient energy.
7. Produces offspring with structural and parameter mutations.

The simulation includes an energy economy, senescence, population management, environmental regimes, CSV logging, and headless-safe visualizations.

## Run the simulation

```bash
python main.py
```

Useful experiments:

```bash
# Longer default-regime run
python main.py --timesteps 10000

# Experimental drifting-food regime
python main.py --forage

# Reproducible headless run
python main.py --seed 7 --no-plots
```

Outputs include population statistics, world visualizations, and history plots. See `config.py` for the simulation parameters that define the energy economy and evolutionary pressure.

## Repository map

| File | Responsibility |
|---|---|
| `organism.py` | Neurons, synapses, threshold propagation, and Hebbian updates |
| `environment.py` | Resource patches, directional sensors, harvesting, and movement |
| `evolution.py` | Replication and structural mutation |
| `simulation.py` | Timestep orchestration and population management |
| `analysis.py` | Sensorimotor, structural, and causal instrumentation |
| `logger.py` | Periodic summary statistics and CSV output |
| `visualizer.py` | World and experiment-history plots |
| `main.py` | Command-line entry point |
| `experiments/` | Experiment configurations and exploratory runs |

## Instrumentation

The simulation exposes three complementary measurements:

### 1. Behavioral: `sensorimotor_alignment`

Cosine similarity between the direction of sensed energy and the direction an organism moved.

- Approximately `0`: movement is uncorrelated with sensed energy
- Greater than `0`: movement tends toward sensed energy
- Less than `0`: movement tends away from sensed energy

### 2. Structural: `frac_with_loop`, `mean_sa_pairs`

These measure whether organisms contain directed sensor-to-actuator paths. They describe the available wiring substrate, not whether the wiring is functionally useful.

### 3. Causal: `matched_reflex_rate`, `mean_influence`

A controlled sensor-activation probe propagates activity through the organism and measures actuator response. This tests whether a sensor drives the actuator associated with the same direction.

## Findings so far

The initial experiments show that structural connectivity does not imply useful behavior.

In the default regime:

- Every organism can contain sensor-to-actuator wiring.
- `sensorimotor_alignment` remains approximately `0.00`.
- `matched_reflex_rate` remains low.

The default economy rewards staying on a resource patch, so there is little selection pressure to develop navigation. The loop exists structurally but is functionally inactive.

In the drifting-food regime, the population currently tends toward extinction rather than evolving navigation. Random networks cannot reliably forage, and the combination of structural mutation and Hebbian adaptation does not discover useful sensorimotor behavior quickly enough.

These are useful negative results: the perception-action loop does not bootstrap automatically under the current mechanisms and timescales.

## Design lessons

The experiments suggest three next directions:

- **Increase learnability:** provide stronger or faster adaptation on sensor-to-actuator pathways.
- **Use curriculum pressure:** transition gradually from stationary resources to drifting resources rather than applying a survival cliff.
- **Measure modularity after behavior emerges:** use influence matrices to test whether subgraphs become associated with specific sensor-to-actuator functions.

## Status

SPECIES is an active research prototype. The current version is focused on making the simulation measurable and falsifiable before adding more complex learning mechanisms.

The most important result so far is methodological: a graph can contain the wiring required for a perception-action loop without exhibiting the function itself.
