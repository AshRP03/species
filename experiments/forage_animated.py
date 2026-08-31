"""Live animated view of the plant ecology + organism population.

Renders the world in a matplotlib window, updating in real time, so you can
watch population dynamics, foraging, crowding, and (once it emerges) navigation
behavior directly instead of inferring it from columns of numbers.

- Green dots = plants, sized by how much energy they currently hold (a big dot
  is a healthy plant; tiny dots are freshly-seeded or nearly grazed out).
- Colored dots = organisms, colored by generation (so you can see lineages
  spread/cluster spatially).
- Title bar shows the same live stats as forage_live.py (pop, plants, gen,
  align, fed%).

Close the window to stop (or Ctrl-C in the terminal).

    cd species/experiments
    python3 forage_animated.py

If no window appears, matplotlib may be defaulting to a headless backend --
try `pip install pyqt5` (or `python3 -m pip install pyqt5`) and re-run.
"""
import argparse
import os
import sys

# Repo root (the `species/` package dir) is the parent of this experiments/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import config

# ------------------------------------------------------------------ #
# Same tuned world as forage_live.py -- keep the two in sync when tuning.
# ------------------------------------------------------------------ #
config.LEARNING_RATE = 0.02             # doubled -- rewards movement toward food more strongly
                                         # by amplifying the existing Hebbian update; see forage_live.py

# --- World carrying-capacity: seed bank (removes absorbing state) + bigger
# --- world (spatial refugia). Both are WORLD FACTS. To isolate the seed bank
# --- alone, revert to: GRID_SIZE=100, PLANT_INIT_COUNT=90, PLANT_MAX_COUNT=380,
# --- SEED_BANK_RATE=1.0.
config.GRID_SIZE = 100                  # was 100 -- 2.25x area; organisms can't sweep it all,
                                         # so ungrazed regions act as refugia (local crashes stay local)
config.SEED_BANK_RATE = 2.0             # spontaneous germination from the dormant soil reservoir,
                                         # independent of living plants -- biomass can never be
                                         # permanently zero, so a crash recovers instead of ending
config.PLANT_ECOLOGY_ENABLED = True
config.PLANT_INIT_COUNT = 200           # scaled from 90 to hold plant density on the bigger world
config.PLANT_MAX_COUNT = 850            # scaled from 380 (density-preserving)
config.PLANT_GROWTH_RATE = 2.0
config.PLANT_ENERGY_MAX = 40.0
config.PLANT_REPRODUCE_PROB = 0.05
config.PLANT_SEED_RADIUS = 5            # was 8 -- see forage_live.py: closes the "colony grows
                                         # onto passive campers" loophole

config.SCENT_DECAY = 6.0                # sharper falloff -> local gradient, not a map-wide sum
config.SCENT_PROBE = 3.0
config.SCENT_NORM = 1.5                 # smaller so a real nearby gradient isn't rounded to ~0

config.HARVEST_RADIUS = 1.5             # was 3.5 -- tighter, forces more precise navigation to
                                         # feed. Movement is confirmed reliable now (multiple
                                         # fixes since the earlier 2.5 instant-extinction attempt).
config.ENERGY_FROM_PATCH = 2.5          # above PLANT_GROWTH_RATE (2.0) -- closes the "camp on
                                         # one plant forever" loophole; see forage_live.py
config.INIT_ENERGY = 30.0
config.SPAWN_RADIUS = 12
config.HUNGER_COST_SCALE = 0.007        # drain accelerates the longer unfed. See forage_live.py
config.METABOLIC_COST = 0.012           # was 0.005 -- faster baseline drain, speeds up pruning
                                         # of the genetic pool. See forage_live.py
config.SENESCENCE_SCALE = 0.002         # was 0.0003 (~10,000-step lethal age -- effectively
                                         # immortal). Aging = metabolic cost * (1 + SCALE*age);
                                         # since income is capped at ENERGY_FROM_PATCH, a high
                                         # enough age ALWAYS becomes unaffordable -> guaranteed
                                         # max lifespan. At 0.002 the lethal age is ~1,500 steps,
                                         # so old organisms turn over instead of persisting and
                                         # crowding. Raise toward 0.003-0.004 for faster turnover
                                         # (shorter lives), lower for longer-lived organisms.
# NOTE: REPRODUCTION_COOLDOWN and MIN_REWARD_BASELINE_FOR_REPRODUCTION were
# DELETED from the engine -- reproduction is energy-gated only now.
config.REPLICATION_THRESHOLD = 100.0     # was 90 -- slightly easier, rewards accurate navigation
config.MAX_POPULATION = 6000

# Ablation condition -- same four conditions as experiments/ablation.py, so you
# can WATCH what each isolates instead of just reading numbers:
#   "baseline"        - both Hebbian learning and mutation active (normal)
#   "mutation_frozen"  - topology/genome never changes across generations;
#                        if navigation still improves, that's Hebbian learning
#   "weights_frozen"   - synapse weights never change within a lifetime; if
#                        navigation still improves across generations, that's
#                        selection acting on inherited structure
#   "both_frozen"      - control; should show no improvement over time at all
# Seed and condition can be overridden from the command line, so you can cycle
# through seeds (seed-robustness check) or conditions without editing the file:
#   python3 forage_animated.py --seed 7
#   python3 forage_animated.py --seed 3 --condition mutation_frozen
_parser = argparse.ArgumentParser(description="Live SPECIES ecology animation")
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument(
    "--condition", default="baseline",
    choices=["baseline", "mutation_frozen", "weights_frozen", "both_frozen"],
)
_args, _ = _parser.parse_known_args()
CONDITION = _args.condition
SEED = _args.seed

if CONDITION in ("mutation_frozen", "both_frozen"):
    config.MICRO_MUTATION_SCALE = 0.0
    config.STRUCTURAL_MUTATION_RATE = 0.0
if CONDITION in ("weights_frozen", "both_frozen"):
    config.HEBBIAN_LEARNING_ENABLED = False

STEPS_PER_FRAME = 1   # sim steps advanced per rendered frame -- lower for a
                        # smoother/slower-motion view, higher to fast-forward

# Color by generations-BEHIND-the-current-leading-edge, not raw generation
# number, with a FIXED scale (set once, never rescaled). Raw generation depth
# drifts in meaning as the max climbs into the hundreds -- the same absolute
# generation gets remapped to a different color every frame. Relative recency
# is stable for the whole run: 0 (the current frontier) is always brightest.
RELATIVE_GEN_WINDOW = 20
NEWBORN_FLASH_FRAMES = 3   # frames a new birth is ringed in red, so it reads as
                            # "just born" instead of being confused with movement
# ------------------------------------------------------------------ #

import simulation

sim = simulation.Simulation(seed=SEED)

fig, ax = plt.subplots(figsize=(8, 8))
plant_scatter = ax.scatter([], [], s=[], c="tab:green", alpha=0.5, label="plants")
org_scatter = ax.scatter(
    [], [], s=16, c=[], cmap="plasma_r", vmin=0, vmax=RELATIVE_GEN_WINDOW,
    label="organisms",
)
fig.colorbar(
    org_scatter, ax=ax,
    label=f"generations behind leading edge (0=newest, {RELATIVE_GEN_WINDOW}+=oldest)",
)
ax.set_xlim(0, sim.grid_size)
ax.set_ylim(0, sim.grid_size)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.legend(loc="upper right")
title = ax.set_title("")

birth_frame: dict[int, int] = {}   # org.id -> frame it was first observed alive
frame_counter = {"n": 0}


def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        if not sim.population:
            break
        sim.step()

    frame_counter["n"] += 1
    fnum = frame_counter["n"]

    env = sim.env
    plant_scatter.set_offsets(env.patch_pos)
    plant_scatter.set_sizes(env.patch_energy * 3 + 2)

    if sim.population:
        pos = np.array([o.position for o in sim.population])
        gens = np.array([o.generation for o in sim.population], dtype=float)
        rel_age = np.clip(gens.max() - gens, 0, RELATIVE_GEN_WINDOW)

        ids = [o.id for o in sim.population]
        for oid in ids:
            if oid not in birth_frame:
                birth_frame[oid] = fnum
        birth_frame_local = {oid: birth_frame[oid] for oid in ids}
        birth_frame.clear()
        birth_frame.update(birth_frame_local)
        is_newborn = np.array(
            [fnum - birth_frame[oid] < NEWBORN_FLASH_FRAMES for oid in ids]
        )

        org_scatter.set_offsets(pos)
        org_scatter.set_array(rel_age)
        edgecolors = np.where(is_newborn, "red", "none")
        linewidths = np.where(is_newborn, 1.4, 0.0)
        org_scatter.set_edgecolors(edgecolors)
        org_scatter.set_linewidths(linewidths)

        align = sim.alignment_window()
        moved = 100.0 * sim.movement_window()
        blind = 100.0 * np.mean([env.sense(o.position).max() < 1e-6 for o in sim.population])
        title.set_text(
            f"[{CONDITION} seed={SEED}]  t={sim.timestep}  pop={len(sim.population)}  "
            f"plants={env.patch_energy.shape[0]}  gen={int(gens.max())}  "
            f"align={align:+.2f}  moved={moved:.0f}%  blind={blind:.0f}%"
        )
    else:
        org_scatter.set_offsets(np.empty((0, 2)))
        title.set_text(f"[{CONDITION} seed={SEED}]  t={sim.timestep}  EXTINCT")

    return plant_scatter, org_scatter, title


anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.show()
