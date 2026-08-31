"""Ablation study: which learning system actually produces rising `align`?

There are two learning engines in this system:
  - Hebbian weight updates within a single organism's lifetime
  - Selection acting on mutated topology/genome traits across generations

This script isolates each by freezing the OTHER one, across multiple seeds per
condition (a crash is a sample, not a failure -- we're looking at distributions
of outcomes, not judging any single run).

Conditions:
  baseline        - both systems active (normal operation)
  mutation_frozen - micro + structural mutation both 0: genome/topology never
                    changes across generations. If align still climbs, Hebbian.
  weights_frozen  - HEBBIAN_LEARNING_ENABLED=False: synapse weights never
                    change within a lifetime. If align still climbs across
                    generations, that's selection on inherited structure.
  both_frozen     - true control. Should show no improvement over time.

    cd species/experiments
    python3 ablation.py
"""
import os
import sys

# Repo root (the `species/` package dir) is the parent of this experiments/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config


def apply_world_config():
    """Same tuned world as forage_live.py -- keep in sync when that changes."""
    config.GRID_SIZE = 150                 # bigger world -> spatial refugia
    config.SEED_BANK_RATE = 2.0            # seed bank -> no permanent absorbing state
    config.PLANT_ECOLOGY_ENABLED = True
    config.PLANT_INIT_COUNT = 200          # density-scaled for the bigger world
    config.PLANT_MAX_COUNT = 850
    config.PLANT_GROWTH_RATE = 2.0
    config.PLANT_ENERGY_MAX = 40.0
    config.PLANT_REPRODUCE_PROB = 0.05
    config.PLANT_SEED_RADIUS = 5
    config.SCENT_DECAY = 6.0
    config.SCENT_PROBE = 3.0
    config.SCENT_NORM = 1.5
    config.HARVEST_RADIUS = 2.0
    config.ENERGY_FROM_PATCH = 2.5
    config.INIT_ENERGY = 30.0
    config.SPAWN_RADIUS = 15
    config.HUNGER_COST_SCALE = 0.003
    config.METABOLIC_COST = 0.007
    config.SENESCENCE_SCALE = 0.0003
    config.REPLICATION_THRESHOLD = 75.0
    config.MAX_POPULATION = 6000
    config.LEARNING_RATE = 0.02


_FROZEN_EVOLUTION = {"MICRO_MUTATION_SCALE": 0.0, "STRUCTURAL_MUTATION_RATE": 0.0}
CONDITIONS = {
    "baseline": {},
    "mutation_frozen": dict(_FROZEN_EVOLUTION),
    "weights_frozen": {"HEBBIAN_LEARNING_ENABLED": False},
    "both_frozen": {**_FROZEN_EVOLUTION, "HEBBIAN_LEARNING_ENABLED": False},
}

SEEDS = [1, 2, 3, 4, 5]
TIMESTEPS = 6000
WINDOW = 2000  # report mean align over each WINDOW-step chunk


def run_one(condition_overrides, seed):
    apply_world_config()
    config.MICRO_MUTATION_SCALE = 0.02  # reset to defaults before applying overrides
    config.STRUCTURAL_MUTATION_RATE = 0.03
    config.HEBBIAN_LEARNING_ENABLED = True
    for k, v in condition_overrides.items():
        setattr(config, k, v)

    import importlib
    import simulation
    importlib.reload(simulation)

    sim = simulation.Simulation(seed=seed)
    window_align: list[float] = []
    accum, n = 0.0, 0
    extinct_at = None

    for t in range(TIMESTEPS):
        if not sim.population:
            extinct_at = sim.timestep
            break
        sim.step()
        a = sim.alignment_window()
        if a != 0.0:
            accum += a
            n += 1
        if (t + 1) % WINDOW == 0:
            window_align.append(accum / n if n else 0.0)
            accum, n = 0.0, 0

    while len(window_align) < TIMESTEPS // WINDOW:
        window_align.append(float("nan"))  # extinct before this window completed

    return {"extinct_at": extinct_at, "windows": window_align}


def main():
    n_windows = TIMESTEPS // WINDOW
    print(f"Ablation study: {len(SEEDS)} seeds x {len(CONDITIONS)} conditions, "
          f"{TIMESTEPS} steps each\n")

    results = {}
    for cond_name, overrides in CONDITIONS.items():
        print(f"=== {cond_name} ({overrides or 'no changes'}) ===")
        trials = []
        for seed in SEEDS:
            r = run_one(overrides, seed)
            trials.append(r)
            windows_str = "  ".join(
                f"{w:+.3f}" if not np.isnan(w) else "  n/a "
                for w in r["windows"]
            )
            ext = f"extinct@{r['extinct_at']}" if r["extinct_at"] is not None else "survived"
            print(f"  seed={seed:>2}  {ext:>16}  align by window: {windows_str}")
        results[cond_name] = trials
        print()

    print("=" * 70)
    print(f"SUMMARY (mean align per {WINDOW}-step window, across {len(SEEDS)} seeds)")
    print("=" * 70)
    header = "condition".ljust(18) + "".join(f"w{i+1}".rjust(9) for i in range(n_windows)) + "  survival"
    print(header)
    for cond_name, trials in results.items():
        window_means = []
        for i in range(n_windows):
            vals = [t["windows"][i] for t in trials if not np.isnan(t["windows"][i])]
            window_means.append(np.mean(vals) if vals else float("nan"))
        survived = sum(1 for t in trials if t["extinct_at"] is None)
        row = cond_name.ljust(18)
        row += "".join(
            f"{w:+.3f}".rjust(9) if not np.isnan(w) else "n/a".rjust(9)
            for w in window_means
        )
        row += f"  {survived}/{len(SEEDS)}"
        print(row)


if __name__ == "__main__":
    main()
