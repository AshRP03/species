"""Live ecology experiment — watch perception-action loops emerge in real time.

World: a living plant ecology (energy sources that grow, spread by seeding, and
die when grazed out), sensed through a directional chemotaxis gradient. Food
never teleports, so tracking is learnable.

Columns:
  pop     - number of organisms
  plants  - number of living plants
  biomass - TOTAL stored energy across all plants (the real food-supply signal;
            plant COUNT can look flat/healthy while biomass underneath collapses
            from grazing, since dying plants are instantly backfilled by seeding
            up to the cap -- this is the fix for that blind spot)
  gen     - deepest lineage generation
  align   - sensorimotor alignment: ~0 = blind, climbing toward +1 = tracking food
  fed%    - fraction of organisms currently on food (bootstrap/crowding health)
  orgE    - mean organism energy

    cd species/experiments
    python3 forage_live.py
"""
import os
import sys

# Repo root (the `species/` package dir) is the parent of this experiments/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config

# ------------------------------------------------------------------ #
# CONFIG OVERRIDES — edit these, then re-run to explore.
# ------------------------------------------------------------------ #
config.LEARNING_RATE = 0.02             # was 0.01 -- doubled. "Reward movement toward food more":
                                         # amplifies the EXISTING local Hebbian/eligibility update
                                         # (stronger reinforcement per co-activation event), rather
                                         # than adding a new explicit alignment-based bonus signal --
                                         # stays within the emergent-economics philosophy (energy
                                         # is still the only reward source, just weighted harder).

# --- World carrying-capacity: seed bank (removes absorbing state) + bigger
# --- world (spatial refugia). Both are WORLD FACTS. To isolate the seed bank
# --- alone, revert to: GRID_SIZE=100, PLANT_INIT_COUNT=90, PLANT_MAX_COUNT=380,
# --- SEED_BANK_RATE=1.0.
config.GRID_SIZE = 150                  # was 100 -- 2.25x area; ungrazed regions act as refugia
config.SEED_BANK_RATE = 2.0             # dormant-soil germination, independent of living plants:
                                         # biomass can never be permanently zero, so crashes recover
config.PLANT_ECOLOGY_ENABLED = True     # living plant ecology (grow / spread / graze-die)
config.PLANT_INIT_COUNT = 200           # scaled from 90 to hold plant density on the bigger world
config.PLANT_MAX_COUNT = 850            # scaled from 380 (density-preserving)
config.PLANT_GROWTH_RATE = 2.0          # robust: a plant sustains ~2 grazers before declining
                                         # -- NOTE: this was set ABOVE a single organism's harvest
                                         # rate (1.0) so a lone founder's plant would be
                                         # self-sustaining, letting the population bootstrap.
                                         # But that also means a stationary organism can now camp
                                         # on one plant forever, never dying, since it can't out-
                                         # eat its own regrowth alone. See ENERGY_FROM_PATCH below.
config.PLANT_ENERGY_MAX = 40.0          # big buffer -> survives transient over-grazing
config.PLANT_REPRODUCE_PROB = 0.05      # how fast plants spread
config.PLANT_SEED_RADIUS = 5            # was 8: a thriving colony spreads a bit less far, so a
                                         # stationary organism sitting near one is less likely to
                                         # have a fresh plant conveniently reseed within its own
                                         # (now tighter) harvest radius -- closes the "the colony
                                         # grows onto passive campers" loophole, distinct from
                                         # the single-plant-camping fix above.

# gradient perception
config.SCENT_DECAY = 6.0                # was 15: sharper falloff -- with many plants on the
                                         # map, a slow decay let the sensed field become an
                                         # aggregate sum dominated by hundreds of distant plants,
                                         # washing out the LOCAL gradient organisms need to climb.
config.SCENT_PROBE = 3.0                # how far ahead each sensor samples
config.SCENT_NORM = 1.5                 # was 4.0: smaller so a real nearby gradient (now a
                                         # smaller raw magnitude thanks to sharper decay) still
                                         # produces a usable signal instead of rounding to ~0.

# organism economy
config.HARVEST_RADIUS = 2.8             # was 3.5. The earlier attempt at 2.5 caused instant
                                         # extinction, but that was BEFORE the motor-babbling fix,
                                         # the shared-actuator-threshold fix, and the hunger clock
                                         # -- movement is now confirmed reliable, so a tighter
                                         # radius should be survivable and forces more precise
                                         # navigation to actually feed, rewarding accurate tracking.
config.ENERGY_FROM_PATCH = 2.5          # was 1.0: NOW ABOVE PLANT_GROWTH_RATE (2.0), so even a
                                         # lone, fully stationary grazer out-eats its own plant's
                                         # regrowth and it declines. Closes the "camp on one plant
                                         # forever" loophole without slowing regrowth/reproduction
                                         # for plants nobody is sitting on -- the "escape" dynamic
                                         # you're seeing should be untouched, but a non-mover's
                                         # own plant should now visibly shrink under them and die
                                         # in ~(40 energy / 0.5 net loss) ~= 80 steps of camping.
config.INIT_ENERGY = 30.0              # founder runway
config.SPAWN_RADIUS = 20                # offspring disperse rather than pile onto one patch
config.HUNGER_COST_SCALE = 0.003        # NEW: drain accelerates the longer an organism goes
                                         # without a successful harvest (extra cost = SCALE *
                                         # steps_since_fed, resets to 0 on any successful feed).
                                         # A brief gap between plants costs almost nothing (e.g.
                                         # 10 steps unfed = +0.03/step, negligible); sustained
                                         # failure compounds fast (100 steps unfed = +0.3/step,
                                         # on top of normal costs; 200 steps = +0.6/step). This
                                         # is the actual "die faster if you can't find food" lever
                                         # -- energy still only ever dies at exactly 0, just gets
                                         # there much quicker for anyone who stops feeding.
config.METABOLIC_COST = 0.007           # was 0.005 -- raised again, moderately. This time it's
                                         # safe to combine with plant scarcity + hunger clock: the
                                         # earlier concern (blanket costs masking the real signal)
                                         # was about diagnosing a broken engine; the engine now
                                         # works, so a faster overall baseline drain just speeds up
                                         # pruning of the genetic pool as requested, on top of the
                                         # scarcity/hunger pressure already doing the differentiating.
config.SENESCENCE_SCALE = 0.0003        # REVERTED from 0.0008, same reasoning.
# NOTE: REPRODUCTION_COOLDOWN and MIN_REWARD_BASELINE_FOR_REPRODUCTION were
# DELETED from the engine (config.py) -- both were hand-designed fitness-shaping
# hacks, not facts about the world. Reproduction is energy-gated only now.
config.REPLICATION_THRESHOLD = 75.0     # was 90 -- slightly easier, so accurate navigation (which
                                         # now costs more to sustain via HARVEST_RADIUS/METABOLIC_COST
                                         # above) converts to reproduction a bit more readily,
                                         # rewarding good foragers rather than making the whole
                                         # population's bar for reproducing harder across the board.
config.MAX_POPULATION = 6000            # safety valve, non-binding; culling-by-weakest-energy
                                         # is the actual capacity enforcement mechanism.

TIMESTEPS = 20000
REPORT_EVERY = 500
SEED = 42
# ------------------------------------------------------------------ #

import simulation

sim = simulation.Simulation(seed=SEED)
print(f"Plant-ecology experiment — seed={SEED}, {TIMESTEPS} steps")
print(
    f"{'t':>6} {'pop':>5} {'plants':>6} {'biomass':>8} {'gen':>4} "
    f"{'align':>7} {'moved%':>6} {'fed%':>5} {'blind%':>6} {'nnDist':>6} {'orgE':>6}",
    flush=True,
)

for t in range(TIMESTEPS):
    if not sim.population:
        print(f"\n** ORGANISMS EXTINCT at t={sim.timestep} **", flush=True)
        break
    sim.step()
    if (t + 1) % REPORT_EVERY == 0:
        pop = sim.population
        pos = np.array([o.position for o in pop], dtype=float)
        pp = sim.env.patch_pos
        if pp.shape[0] == 0:
            nearest = np.full(len(pop), np.inf)  # total plant extinction -- nothing is "near"
        else:
            nearest = np.sqrt(((pos[:, None, :] - pp[None, :, :]) ** 2).sum(2)).min(1)
        fed = 100.0 * np.mean(nearest <= config.HARVEST_RADIUS)
        org_e = np.mean([o.energy for o in pop])
        biomass = float(sim.env.patch_energy.sum())
        align = sim.alignment_window()
        moved = 100.0 * sim.movement_window()
        # blind% = fraction with ZERO sensor signal in all 4 directions right now
        # -- i.e. no plant within SENSOR_RANGE at all. Distinguishes "wandered
        # somewhere with nothing to sense" (spatial/scarcity problem, memory
        # can't help) from "had signal, didn't act on it" (behavioural problem,
        # what the memory fix targets).
        blind = 100.0 * np.mean([sim.env.sense(o.position).max() < 1e-6 for o in pop])
        # nnDist = mean nearest-OTHER-ORGANISM distance -- the dispersal index.
        # Watch this against Phase 1 social sensing: does it change (organisms
        # actively spreading out to avoid crowding) now that they can sense
        # each other, versus being purely a byproduct of where food happens to be?
        if len(pop) > 1:
            org_dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(2))
            np.fill_diagonal(org_dist, np.inf)
            nn_dist = float(org_dist.min(axis=1).mean())
        else:
            nn_dist = 0.0
        print(
            f"{sim.timestep:>6} {len(pop):>5} {sim.env.patch_energy.shape[0]:>6} "
            f"{biomass:>8.0f} {max(o.generation for o in pop):>4} "
            f"{align:>+7.3f} {moved:>6.0f} {fed:>5.0f} {blind:>6.0f} {nn_dist:>6.1f} {org_e:>6.1f}",
            flush=True,
        )
else:
    print(f"\nSURVIVED to t={sim.timestep} with {len(sim.population)} organisms", flush=True)
