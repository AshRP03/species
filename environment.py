"""The 2D world: a field of regenerating energy patches.

The environment is the only teacher. It exposes three things to organisms:
  - sense(pos)   -> 4 directional energy-proximity signals (N/S/E/W) in [0, 1]
  - harvest(pos) -> energy drawn from a nearby patch (depletes that patch)
  - regenerate_patches() -> patches slowly refill each timestep

Everything is vectorized over patches with numpy so the per-organism cost stays
low even at MAX_POPULATION.
"""

from __future__ import annotations

import numpy as np

import config

# Direction order is fixed as N, S, E, W everywhere (sensors and actuators).
# Screen convention: +x = East, +y = South.
_DIRS = np.array(
    [
        [0.0, -1.0],   # N
        [0.0, 1.0],    # S
        [1.0, 0.0],    # E
        [-1.0, 0.0],   # W
    ]
)



class Environment:
    def __init__(self, rng: np.random.Generator, grid_size: int = config.GRID_SIZE):
        self.grid_size = grid_size
        self.rng = rng
        if config.PLANT_ECOLOGY_ENABLED:
            n = config.PLANT_INIT_COUNT
            self.patch_pos = rng.integers(0, grid_size, size=(n, 2)).astype(np.float64)
            self.patch_energy = rng.uniform(
                0.5 * config.PLANT_ENERGY_MAX, config.PLANT_ENERGY_MAX, size=n
            )
        else:
            n = config.ENERGY_PATCH_COUNT
            self.patch_pos = rng.integers(0, grid_size, size=(n, 2)).astype(np.float64)
            # Start patches partially full so founders have something to find.
            self.patch_energy = rng.uniform(
                0.5 * config.ENERGY_PATCH_MAX, config.ENERGY_PATCH_MAX, size=n
            )
        self.patch_age = np.zeros(n)  # steps since this bloom (re)spawned (bloom mode only)

    # -- sensing -----------------------------------------------------------
    def sense(self, position: tuple[int, int]) -> np.ndarray:
        """Return 5 sensor values in [0, 1): 4 directional gradients (N,S,E,W),
        plus 1 non-directional "local field strength" reading.

        Each sample point reads a smooth concentration field:
        C(x) = sum_i energy_i * exp(-dist / SCENT_DECAY). The 4 directional
        sensors report the *gradient* -- how much stronger the field is one
        probe-step ahead than right here -- which is what makes approach
        learnable ("warmer / colder"). But that gradient collapses to ~0 right
        AT a plant (nothing nearby looks better than here), which is exactly
        the moment an organism most needs a "stay, this is good" signal, not
        silence. The 5th channel fixes that blind spot: it's the RAW field
        value at the organism's current position (not a difference), so it
        stays strong for as long as the organism is actually near food,
        independent of direction -- a "graze here" signal a network can learn
        to use to resist babbling itself back out of a good patch.
        """
        out = np.zeros(5, dtype=np.float64)
        if self.patch_pos.shape[0] == 0:
            return out
        pos = np.asarray(position, dtype=np.float64)
        # Field at the organism plus one probe-step ahead in each direction.
        points = np.vstack([pos[None, :], pos + config.SCENT_PROBE * _DIRS])  # (5, 2)
        diff = points[:, None, :] - self.patch_pos[None, :, :]                # (5, P, 2)
        dist = np.sqrt((diff * diff).sum(axis=2))                             # (5, P)
        C = (self.patch_energy[None, :] * np.exp(-dist / config.SCENT_DECAY)).sum(axis=1)
        gradient = (C[1:] - C[0]) / config.SCENT_NORM        # (4,) ahead minus here
        local_strength = 1.0 - np.exp(-C[0] / config.SCENT_LOCAL_NORM)  # (1,) raw field, here
        out[:4] = np.clip(gradient, 0.0, 1.0)
        out[4] = local_strength
        return out

    # -- harvesting --------------------------------------------------------
    def harvest(self, position: tuple[int, int]) -> float:
        """Draw energy from the nearest patch within HARVEST_RADIUS and deplete it."""
        if self.patch_pos.shape[0] == 0:
            return 0.0  # total plant extinction -- nothing to harvest, nothing crashes
        pos = np.asarray(position, dtype=np.float64)
        dist = np.linalg.norm(self.patch_pos - pos, axis=1)
        idx = int(np.argmin(dist))
        if dist[idx] > config.HARVEST_RADIUS:
            return 0.0
        gain = min(config.ENERGY_FROM_PATCH, float(self.patch_energy[idx]))
        self.patch_energy[idx] -= gain
        # In the foraging regime, an exhausted patch regrows somewhere new, so
        # organisms can't camp — they must track food as it moves.
        if config.PATCH_RELOCATE_ON_DEPLETION and self.patch_energy[idx] <= 1e-6:
            r = config.PATCH_RELOCATE_RADIUS
            if r > 0:  # local drift: food reappears nearby, followable by a navigator
                offset = self.rng.integers(-r, r + 1, size=2)
                new = np.clip(self.patch_pos[idx] + offset, 0, self.grid_size - 1)
            else:      # global teleport
                new = self.rng.integers(0, self.grid_size, size=2)
            self.patch_pos[idx] = new.astype(np.float64)
            self.patch_energy[idx] = config.ENERGY_PATCH_MAX * 0.5
        return gain

    # -- dynamics ----------------------------------------------------------
    def _respawn(self, idx: int) -> None:
        """A bloom despawns and a fresh one appears (keeps food supply constant)."""
        r = config.PATCH_RESPAWN_RADIUS
        if r > 0:  # reappears nearby — followable
            offset = self.rng.integers(-r, r + 1, size=2)
            new = np.clip(self.patch_pos[idx] + offset, 0, self.grid_size - 1)
        else:      # reappears anywhere — must be searched for
            new = self.rng.integers(0, self.grid_size, size=2)
        self.patch_pos[idx] = new.astype(np.float64)
        self.patch_energy[idx] = config.ENERGY_PATCH_MAX * 0.5
        self.patch_age[idx] = 0

    def _effective_max_age(self, timestep: int) -> float:
        """Curriculum: blooms are near-permanent early, then lifespan shrinks to
        PATCH_MAX_AGE, ramping foraging pressure in gradually."""
        c = config.FORAGE_CURRICULUM_STEPS
        f = 1.0 if c <= 0 else min(1.0, timestep / c)
        # (1-f) adds up to ~5000 steps of extra life early on -> effectively static.
        return config.PATCH_MAX_AGE + (1.0 - f) * 5000.0

    def _step_plants(self) -> None:
        """Living-ecology dynamics: plants die if grazed out, survivors grow, and
        mature plants seed nearby. The food *pattern* moves as plants collapse
        where organisms cluster and thrive at the ungrazed seeding frontier."""
        # Death first (on post-harvest energy): heavy grazing below the threshold
        # kills a plant. Checking before growth means a fully-grazed plant can't
        # be saved by this step's photosynthesis.
        alive = self.patch_energy >= config.PLANT_DEATH_ENERGY
        if not alive.all():
            self.patch_pos = self.patch_pos[alive]
            self.patch_energy = self.patch_energy[alive]

        # Growth + local reproduction only apply to plants that currently exist.
        # (No early return on zero -- the seed bank below MUST run even when the
        # ecology has been grazed to nothing, since that's exactly the case it
        # exists to recover from.)
        if self.patch_energy.shape[0] > 0:
            # Growth (photosynthesis), capped.
            self.patch_energy = np.minimum(
                self.patch_energy + config.PLANT_GROWTH_RATE, config.PLANT_ENERGY_MAX
            )

            # Reproduction: mature plants always attempt to seed a nearby
            # offspring, paying its energy. PLANT_MAX_COUNT is enforced by
            # culling the weakest afterward, not by blocking reproduction
            # outright (a hard block there silently freezes reproduction at the
            # cap -- the same bug we fixed for organisms).
            mature = np.nonzero(self.patch_energy >= config.PLANT_REPRODUCE_THRESHOLD)[0]
            seeds_pos: list[np.ndarray] = []
            seeds_e: list[float] = []
            for i in mature:
                if self.rng.random() < config.PLANT_REPRODUCE_PROB:
                    offset = self.rng.integers(
                        -config.PLANT_SEED_RADIUS, config.PLANT_SEED_RADIUS + 1, size=2
                    )
                    p = np.clip(self.patch_pos[i] + offset, 0, self.grid_size - 1)
                    seeds_pos.append(p.astype(np.float64))
                    seeds_e.append(config.PLANT_SEED_ENERGY)
                    self.patch_energy[i] -= config.PLANT_SEED_ENERGY
            if seeds_pos:
                self.patch_pos = np.vstack([self.patch_pos, np.array(seeds_pos)])
                self.patch_energy = np.concatenate([self.patch_energy, np.array(seeds_e)])

        # Seed bank: spontaneous germination from the dormant soil reservoir,
        # independent of any living plant. Runs unconditionally (even at zero
        # plants), which is what makes total collapse recoverable rather than
        # permanent. New sprouts land anywhere -- largely in ungrazed space --
        # and grow into colony founders via the local reproduction above.
        if config.SEED_BANK_RATE > 0.0:
            k = int(self.rng.poisson(config.SEED_BANK_RATE))
            if k > 0:
                new_pos = self.rng.integers(0, self.grid_size, size=(k, 2)).astype(np.float64)
                new_e = np.full(k, config.PLANT_SEED_ENERGY, dtype=np.float64)
                self.patch_pos = np.vstack([self.patch_pos, new_pos])
                self.patch_energy = np.concatenate([self.patch_energy, new_e])

        overflow = self.patch_energy.shape[0] - config.PLANT_MAX_COUNT
        if overflow > 0:
            order = np.argsort(self.patch_energy)
            keep = order[overflow:]
            self.patch_pos = self.patch_pos[keep]
            self.patch_energy = self.patch_energy[keep]

    def regenerate_patches(self, timestep: int = 0) -> None:
        if config.PLANT_ECOLOGY_ENABLED:
            self._step_plants()
            return
        self.patch_energy = np.minimum(
            self.patch_energy + config.ENERGY_PATCH_REGEN_RATE, config.ENERGY_PATCH_MAX
        )
        # Continuous drift: food slowly wanders, so a static "camper" gradually
        # loses its patch while an organism that tracks sensed energy keeps up.
        if config.PATCH_DRIFT_STEP > 0.0:
            step = self.rng.normal(0.0, config.PATCH_DRIFT_STEP, size=self.patch_pos.shape)
            self.patch_pos = np.clip(self.patch_pos + step, 0, self.grid_size - 1)

        # Dynamic blooms: age patches, despawn+respawn those that expire or deplete.
        if config.FORAGE_ENABLED:
            self.patch_age += 1
            eff_age = self._effective_max_age(timestep)
            expired = (self.patch_age >= eff_age) | (self.patch_energy <= 1e-6)
            for idx in np.nonzero(expired)[0]:
                self._respawn(int(idx))


def apply_movement(actuator_states: list[float]) -> tuple[int, int]:
    """Turn the 4 actuator states (N, S, E, W order) into a (dx, dy) step.

    Opposing actuators cancel; the organism moves at most one cell per axis.
    """
    n, s, e, w = actuator_states
    dx = int(np.sign(e - w))
    dy = int(np.sign(s - n))
    return dx, dy
