"""Population management and the per-timestep orchestration.

One timestep, per organism:
  1. sense the environment (sets sensor neuron states)
  2. propagate one threshold pass
  3. act: actuator states drive a move to a new cell
  4. harvest energy at the new cell, tally metabolic/firing/edge costs
  5. local Hebbian update gated by the sign of the net energy change
  6. apply the energy delta, age, then possibly die or replicate
Then the environment regenerates and the population lists are reconciled.
"""

from __future__ import annotations

import numpy as np

import config
from analysis import alignment, sensor_vector
from environment import Environment, apply_movement
from evolution import maybe_replicate
from organism import Organism, make_random_organism, propagate, update_weights


class Simulation:
    def __init__(self, seed: int = config.SEED, grid_size: int = config.GRID_SIZE):
        self.rng = np.random.default_rng(seed)
        self.grid_size = grid_size
        self.env = Environment(self.rng, grid_size)
        self._next_id = 0
        self.population: list[Organism] = [
            make_random_organism(self._alloc_id(), self.rng, grid_size)
            for _ in range(config.INIT_POPULATION)
        ]
        self.timestep = 0
        self.mean_alignment = 0.0

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # -- per-organism update ----------------------------------------------
    def _set_sensors(self, org: Organism):
        readings = self.env.sense(org.position)
        for sensor_id, value in zip(org.sensor_ids, readings):
            org.neurons[sensor_id].state = float(value)
        return readings

    def _act(self, org: Organism) -> tuple[int, int]:
        actuator_states = [org.neurons[a].state for a in org.actuator_ids]
        # Pad in case a mutation ever changed the actuator count.
        while len(actuator_states) < 4:
            actuator_states.append(0.0)
        dx, dy = apply_movement(actuator_states[:4])
        x = min(self.grid_size - 1, max(0, org.position[0] + dx))
        y = min(self.grid_size - 1, max(0, org.position[1] + dy))
        org.position = (x, y)
        return dx, dy

    def _energy_delta(self, org: Organism, patch_gain: float) -> float:
        senescence = 1.0 + config.SENESCENCE_SCALE * org.age
        node_cost = len(org.neurons) * config.METABOLIC_COST * senescence
        fire_cost = sum(n.state for n in org.neurons.values()) * config.FIRING_COST
        edge_cost = sum(abs(s.weight) for s in org.synapses) * config.EDGE_COST_SCALE
        return patch_gain - node_cost - fire_cost - edge_cost

    # -- one world timestep -----------------------------------------------
    def step(self) -> None:
        dead: list[Organism] = []
        newborns: list[Organism] = []
        capacity = config.MAX_POPULATION - len(self.population)

        align_sum = 0.0
        align_count = 0

        for org in self.population:
            readings = self._set_sensors(org)
            propagate(org)
            dx, dy = self._act(org)
            # Behavioural perception-action signal: did it move toward energy?
            a = alignment(sensor_vector(readings), np.array([dx, dy], dtype=float))
            if a is not None:
                align_sum += a
                align_count += 1
            patch_gain = self.env.harvest(org.position)
            delta = self._energy_delta(org, patch_gain)
            update_weights(org, delta)
            org.energy += delta
            org.age += 1

            if org.energy <= 0.0:
                dead.append(org)
                continue

            if len(newborns) < capacity:
                child = maybe_replicate(org, self.rng, self._next_id, self.grid_size)
                if child is not None:
                    self._next_id += 1
                    newborns.append(child)

        self.env.regenerate_patches()
        # Mean sensorimotor alignment this step (organisms that both sensed and
        # moved). ~0 = blind wandering; > 0 = moving toward sensed energy.
        self.mean_alignment = align_sum / align_count if align_count else 0.0

        if dead:
            dead_ids = {id(o) for o in dead}
            self.population = [o for o in self.population if id(o) not in dead_ids]
        self.population.extend(newborns)
        self.timestep += 1

    def run(self, timesteps: int, on_step=None) -> None:
        for _ in range(timesteps):
            if not self.population:
                break
            self.step()
            if on_step is not None:
                on_step(self)
