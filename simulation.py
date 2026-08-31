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
        self._align_accum = 0.0
        self._align_n = 0
        self._moved_accum = 0.0
        self._moved_n = 0

    def alignment_window(self) -> float:
        """Mean sensorimotor alignment since the last call, then reset. Smoother
        than the single-step value; used by the logger for a readable trend."""
        m = self._align_accum / self._align_n if self._align_n else 0.0
        self._align_accum = 0.0
        self._align_n = 0
        return m

    def movement_window(self) -> float:
        """Fraction of organism-steps with nonzero displacement since the last
        call, then reset. Distinguishes "not moving at all" from "moving but
        uncorrelated with sensed food" -- align alone can't tell those apart,
        since it only counts steps where BOTH sensing and movement were nonzero."""
        m = self._moved_accum / self._moved_n if self._moved_n else 0.0
        self._moved_accum = 0.0
        self._moved_n = 0
        return m

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # -- per-organism update ----------------------------------------------
    def _social_density(self) -> np.ndarray:
        """Local conspecific density for every organism, all at once (vectorized
        pairwise field, same math as the food local_strength sensor). Computed
        once per step from positions BEFORE anyone moves this tick, so sensing
        is synchronous and doesn't depend on loop order within the step."""
        n = len(self.population)
        if n == 0:
            return np.zeros(0)
        pos = np.array([o.position for o in self.population], dtype=np.float64)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        field = np.exp(-dist / config.SOCIAL_DECAY).sum(axis=1) - 1.0  # exclude self (dist=0 -> 1.0)
        return 1.0 - np.exp(-np.maximum(field, 0.0) / config.SOCIAL_LOCAL_NORM)

    def _set_sensors(self, org: Organism, social_reading: float):
        env_readings = self.env.sense(org.position)
        readings = np.concatenate([env_readings, [social_reading]])
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
        # Hunger clock: an extra cost that GROWS the longer an organism has gone
        # without a successful harvest, so drain accelerates over time instead of
        # staying flat -- a brief gap between plants costs almost nothing, but
        # sustained failure to feed compounds toward death quickly. Resets to 0
        # on any step with a successful harvest (see step()).
        hunger_cost = config.HUNGER_COST_SCALE * org.steps_since_fed
        return patch_gain - node_cost - fire_cost - edge_cost - hunger_cost

    # -- one world timestep -----------------------------------------------
    def step(self) -> None:
        dead: list[Organism] = []
        newborns: list[Organism] = []

        align_sum = 0.0
        align_count = 0
        moved_sum = 0.0
        social = self._social_density()

        for org, social_reading in zip(self.population, social):
            readings = self._set_sensors(org, social_reading)
            propagate(org, self.rng)
            dx, dy = self._act(org)
            # Raw movement rate: did it move AT ALL, regardless of direction.
            # Isolates "not moving" from "moving but poorly aligned" -- align
            # alone conflates the two, since it only counts steps where both
            # sensing and movement were nonzero.
            if dx != 0 or dy != 0:
                moved_sum += 1.0
            # Behavioural perception-action signal: did it move toward energy?
            a = alignment(sensor_vector(readings), np.array([dx, dy], dtype=float))
            if a is not None:
                align_sum += a
                align_count += 1
            patch_gain = self.env.harvest(org.position)
            org.steps_since_fed = 0 if patch_gain > 0.0 else org.steps_since_fed + 1
            delta = self._energy_delta(org, patch_gain)
            if config.HEBBIAN_LEARNING_ENABLED:
                update_weights(org, delta)
            org.energy += delta
            org.age += 1

            if org.energy <= 0.0:
                dead.append(org)
                continue

            # Always attempt replication when eligible -- capacity is enforced by
            # culling the weakest afterward, not by blocking reproduction outright.
            # Blocking outright (the old behaviour) freezes evolution entirely once
            # the cap is hit: no deaths -> no capacity -> no births -> no turnover.
            child = maybe_replicate(org, self.rng, self._next_id, self.grid_size)
            if child is not None:
                self._next_id += 1
                newborns.append(child)

        self.env.regenerate_patches(self.timestep)
        # Mean sensorimotor alignment this step (organisms that both sensed and
        # moved). ~0 = blind wandering; > 0 = moving toward sensed energy.
        self.mean_alignment = align_sum / align_count if align_count else 0.0
        # Accumulate for an interval-averaged alignment (per-step value is noisy).
        self._align_accum += align_sum
        self._align_n += align_count
        # Accumulate raw movement rate over ALL organism-steps this tick (the
        # denominator here is population size, unlike align's stricter count).
        self._moved_accum += moved_sum
        self._moved_n += len(self.population)

        if dead:
            dead_ids = {id(o) for o in dead}
            self.population = [o for o in self.population if id(o) not in dead_ids]
        self.population.extend(newborns)

        # Enforce carrying capacity by culling the weakest (lowest-energy)
        # organisms, so a saturated population still turns over generationally
        # instead of freezing -- births continue; the cost falls on whoever is
        # closest to starving anyway, which is how carrying capacity actually
        # works in biology.
        overflow = len(self.population) - config.MAX_POPULATION
        if overflow > 0:
            self.population.sort(key=lambda o: o.energy)
            del self.population[:overflow]

        self.timestep += 1

    def run(self, timesteps: int, on_step=None) -> None:
        for _ in range(timesteps):
            if not self.population:
                break
            self.step()
            if on_step is not None:
                on_step(self)
