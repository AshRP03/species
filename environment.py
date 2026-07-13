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

HARVEST_RADIUS = 2.0  # cells; organism eats from any patch within this distance


class Environment:
    def __init__(self, rng: np.random.Generator, grid_size: int = config.GRID_SIZE):
        self.grid_size = grid_size
        self.rng = rng
        n = config.ENERGY_PATCH_COUNT
        self.patch_pos = rng.integers(0, grid_size, size=(n, 2)).astype(np.float64)
        # Start patches partially full so founders have something to find.
        self.patch_energy = rng.uniform(
            0.5 * config.ENERGY_PATCH_MAX, config.ENERGY_PATCH_MAX, size=n
        )

    # -- sensing -----------------------------------------------------------
    def sense(self, position: tuple[int, int]) -> np.ndarray:
        """Return 4 sensor values in [0, 1] for N, S, E, W.

        Each patch within SENSOR_RANGE contributes to a direction proportionally
        to how much of its energy is stored, how aligned it is with that
        compass direction, and how close it is.
        """
        pos = np.asarray(position, dtype=np.float64)
        delta = self.patch_pos - pos                     # (P, 2) vector to each patch
        dist = np.linalg.norm(delta, axis=1)             # (P,)
        in_range = dist < config.SENSOR_RANGE
        out = np.zeros(4, dtype=np.float64)
        if not np.any(in_range):
            return out

        delta = delta[in_range]
        dist = dist[in_range]
        energy = self.patch_energy[in_range]
        # Unit vectors toward each in-range patch (avoid div-by-zero when on it).
        safe = np.maximum(dist, 1e-6)
        unit = delta / safe[:, None]                     # (Q, 2)
        alignment = np.clip(unit @ _DIRS.T, 0.0, None)   # (Q, 4), only forward hemisphere
        proximity = 1.0 - dist / config.SENSOR_RANGE     # (Q,) closer -> stronger
        strength = (energy / config.ENERGY_PATCH_MAX) * proximity
        out = (alignment * strength[:, None]).sum(axis=0)
        return np.clip(out, 0.0, 1.0)

    # -- harvesting --------------------------------------------------------
    def harvest(self, position: tuple[int, int]) -> float:
        """Draw energy from the nearest patch within HARVEST_RADIUS and deplete it."""
        pos = np.asarray(position, dtype=np.float64)
        dist = np.linalg.norm(self.patch_pos - pos, axis=1)
        idx = int(np.argmin(dist))
        if dist[idx] > HARVEST_RADIUS:
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
    def regenerate_patches(self) -> None:
        self.patch_energy = np.minimum(
            self.patch_energy + config.ENERGY_PATCH_REGEN_RATE, config.ENERGY_PATCH_MAX
        )
        # Continuous drift: food slowly wanders, so a static "camper" gradually
        # loses its patch while an organism that tracks sensed energy keeps up.
        if config.PATCH_DRIFT_STEP > 0.0:
            step = self.rng.normal(0.0, config.PATCH_DRIFT_STEP, size=self.patch_pos.shape)
            self.patch_pos = np.clip(self.patch_pos + step, 0, self.grid_size - 1)


def apply_movement(actuator_states: list[float]) -> tuple[int, int]:
    """Turn the 4 actuator states (N, S, E, W order) into a (dx, dy) step.

    Opposing actuators cancel; the organism moves at most one cell per axis.
    """
    n, s, e, w = actuator_states
    dx = int(np.sign(e - w))
    dy = int(np.sign(s - n))
    return dx, dy
