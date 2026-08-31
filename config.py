"""All simulation constants in one place.

These are the knobs that define the energy economy and the pressures acting on
the population. V0 values come straight from the concept brief; tune here rather
than sprinkling magic numbers through the code.
"""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
GRID_SIZE = 100                  # world is GRID_SIZE x GRID_SIZE cells
ENERGY_PATCH_COUNT = 100
ENERGY_PATCH_REGEN_RATE = 1.0    # energy units added to each patch per timestep
ENERGY_PATCH_MAX = 50.0          # cap on stored energy per patch (renewable "grass")
SENSOR_RANGE = 30.0              # how far a sensor "sees" energy (cells)
HARVEST_RADIUS = 2.0             # cells; organism eats from a patch within this distance

# ---------------------------------------------------------------------------
# Gradient ("scent") perception
# ---------------------------------------------------------------------------
# Sensors read a smooth concentration field C(x) = sum_i energy_i*exp(-dist/decay)
# sampled a short distance ahead in each compass direction, so moving toward food
# monotonically raises the signal — a chemotaxis gradient the organism can climb.
SCENT_DECAY = 15.0               # spatial falloff of the field (cells)
SCENT_PROBE = 3.0                # how far ahead each sensor samples the field (cells)
SCENT_NORM = 4.0                 # scales the ahead-minus-here gradient into [0, 1]
# 5th, non-directional sensor: raw field strength AT the organism's current
# position (not a difference). The directional gradient goes quiet exactly at
# a plant (nothing nearby looks "better" than here); this channel stays strong
# instead, giving a learnable "graze here, don't leave" signal. Tuned so it
# saturates toward 1.0 when standing on/near a well-fed plant (PLANT_ENERGY_MAX
# ~40) and falls off within roughly SCENT_DECAY cells.
SCENT_LOCAL_NORM = 15.0

# 6th sensor (Phase 1 of inter-organism sensing): passive awareness of nearby
# conspecifics -- how crowded is it here, right now. Same field-and-squash
# pattern as SCENT_LOCAL_NORM above, just summed over nearby organisms instead
# of nearby plants. No emission/signaling yet -- this is purely "can I tell
# others are near me," the substrate a later communication layer would need.
SOCIAL_DECAY = 10.0               # spatial falloff of the conspecific field (cells)
SOCIAL_LOCAL_NORM = 3.0           # scales local organism density into [0, 1)

# ---------------------------------------------------------------------------
# Plant ecology (energy sources that grow, spread by seeding, and die if grazed
# out). Nothing teleports: the *pattern* of food moves as plants thrive where
# ungrazed and die where organisms cluster. Enable via PLANT_ECOLOGY_ENABLED.
# ---------------------------------------------------------------------------
PLANT_ECOLOGY_ENABLED = False
PLANT_INIT_COUNT = 60            # founder plants
# High safety valve, not a designed pressure -- plants should be limited by grazing
# and physical space, not an arbitrary count ceiling. A LOW cap here reproduces the
# exact reproduction-freeze bug we fixed for organisms: once plants hit the cap,
# new seeding is blocked outright, so the count looks artificially "stable" at the
# ceiling while local biomass can still be collapsing underneath (masked by the
# flat count). Keep this comfortably above whatever organism population reaches.
PLANT_MAX_COUNT = 3000
PLANT_ENERGY_MAX = 20.0          # cap on a single plant's stored energy
PLANT_GROWTH_RATE = 0.4          # energy gained per plant per step (photosynthesis)
PLANT_REPRODUCE_THRESHOLD = 12.0 # energy above which a plant may seed
PLANT_REPRODUCE_PROB = 0.03      # per-step chance a mature plant seeds
PLANT_SEED_ENERGY = 4.0          # newborn plant's energy (taken from the parent)
PLANT_SEED_RADIUS = 8            # offspring appears within this radius of parent
PLANT_DEATH_ENERGY = 0.4         # grazed below this -> the plant dies
# Seed bank: expected spontaneous germinations per timestep from a dormant soil
# reservoir, INDEPENDENT of any living plants. This removes the "biomass = 0 is
# permanent" absorbing state -- local reproduction (above) needs a living parent
# plant, so once organisms graze everything to zero the ecology can never
# recover; the seed bank keeps a trickle of new plants germinating at random
# regardless, so total collapse becomes a recoverable boom-bust instead of the
# end. A WORLD FACT (there are always dormant seeds in the soil), not a fitness
# hack. Scale with world area. 0.0 = disabled (old absorbing-state behaviour).
SEED_BANK_RATE = 0.0

# ---------------------------------------------------------------------------
# Organism
# ---------------------------------------------------------------------------
INIT_POPULATION = 50
INIT_NODES = 40                  # interneurons + sensors + actuators at spawn
INIT_EDGE_PROB = 0.05            # sparse random graph
SENSOR_COUNT = 6                 # N/S/E/W directional gradient + local food strength + local
                                  # conspecific density (Phase 1 social sensing)
ACTUATOR_COUNT = 4               # N / S / E / W movement drive
INIT_ENERGY = 20.0               # energy a founder organism starts with (survival runway)
INIT_WEIGHT_SCALE = 0.5          # std of initial synapse weights
# "Motor babbling" for actuators: no fixed bias (a fixed value drawn once at
# birth can only ever settle into permanently-on or permanently-off -- it can't
# sustain ongoing exploration). Instead, actuators get FRESH random noise added
# to their weighted input every single timestep (see organism.propagate()),
# plus an easy-to-clear threshold, so there's a real, ONGOING, non-fizzling
# chance to fire -- giving Hebbian learning continuous co-activation events to
# shape into purposeful movement. Every actuator is also guaranteed at least
# one incoming edge at creation (see make_random_organism()), so none can ever
# be structurally stranded with zero possible input.
#
# GENOME, NOT CONSTANT: threshold, memory decay, and noise scale below are only
# the FOUNDER initialization ranges. Every neuron's own threshold/decay/noise
# is a heritable, mutable trait from here on (see organism.Neuron.memory_decay
# / .noise_scale, and evolution._micro_mutate) -- selection, not a
# hand-picked constant, decides where these end up for any given lineage. This
# replaced two hacks: actuators sharing one forced-equal threshold (an
# uncorrectable-by-design workaround for the fact that thresholds used to never
# change after birth), and a single global noise magnitude for every actuator
# everywhere. Now an organism can evolve from noisy exploration toward
# deterministic control as its wiring becomes reliable, and opposing actuators
# can converge OR differentiate, whichever selection favours.
ACTUATOR_THRESHOLD_RANGE = (0.1, 0.4)         # founder init range only
ACTUATOR_NOISE_SCALE_RANGE = (0.05, 0.4)      # founder init range only
INTERNEURON_MEMORY_DECAY_RANGE = (0.0, 0.95)  # founder init range only

# ---------------------------------------------------------------------------
# Energy costs / gains -- FROZEN. These are "world physics" (facts about the
# environment), not organism decisions -- they don't belong in the genome, but
# they also don't belong getting re-tuned after every run. Repeatedly nudging
# these was masking the real bottleneck by shifting WHEN/HOW populations died
# rather than addressing WHY. Pick values once, hold them, and let variation in
# outcomes come from the organisms (genome/mutation/selection), not from us
# quietly re-balancing the world underneath them.
# ---------------------------------------------------------------------------
METABOLIC_COST = 0.005           # per node per timestep (cost of simply existing)
# Senescence: an extension beyond the brief's starvation-only death. Metabolic
# cost grows with age so long-lived "campers" eventually die, forcing population
# turnover. Without this the population freezes into an immortal subsistence pool
# and evolution stalls. Set to 0.0 to recover the brief's original behavior.
SENESCENCE_SCALE = 0.0004        # metabolic multiplier = 1 + SENESCENCE_SCALE * age
FIRING_COST = 0.02               # per spike emitted
EDGE_COST_SCALE = 0.001          # cost proportional to |weight| summed over edges
ENERGY_FROM_PATCH = 1.0          # max energy harvested per timestep while on a patch
# Hunger clock: extra cost = HUNGER_COST_SCALE * steps_since_fed, so drain
# ACCELERATES the longer an organism goes without a successful harvest, rather
# than staying flat. Resets to 0 on any step that harvests > 0. At 0.0 this
# recovers today's behavior (no acceleration).
HUNGER_COST_SCALE = 0.0

# ---------------------------------------------------------------------------
# Local learning (Hebbian + energy gate)
# ---------------------------------------------------------------------------
LEARNING_RATE = 0.01
WEIGHT_PRUNE_THRESHOLD = 0.02    # synapses weaker than this are removed
WEIGHT_CLIP = 5.0                # keep weights bounded for numerical sanity
# Eligibility traces + reward baseline (temporal credit assignment). A synapse
# keeps a decaying memory of recent pre*post coincidence; when the energy
# neuromodulator arrives it reinforces recently-active synapses, so the approach
# that *led* to food gets credited, not just the instant of arrival. The
# modulator is the advantage of this step over the organism's own recent
# baseline, so improving toward food reinforces even while net energy is < 0.
ELIGIBILITY_DECAY = 0.9          # trace memory; time constant ~1/(1-decay) steps
REWARD_BASELINE_RATE = 0.05      # EMA rate for the per-organism reward baseline
# Ablation switch: set False to freeze ALL within-lifetime weight learning
# (update_weights becomes a no-op; weights stay wherever birth/inheritance left
# them). Combine with MICRO_MUTATION_SCALE=0 and STRUCTURAL_MUTATION_RATE=0
# (freezes genome/topology evolution instead) to isolate which of the two
# learning systems -- Hebbian plasticity within a life, or selection on
# structure across lives -- is actually responsible for any observed
# improvement in behaviour over time.
HEBBIAN_LEARNING_ENABLED = True

# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------
REPLICATION_THRESHOLD = 30.0     # energy needed to spawn a child
# Reproduction gate is ENERGY ONLY now. Two previous gates here were deleted as
# hand-designed fitness-shaping, not facts about the world:
#   - a flat wall-clock cooldown between births (not biological; was the
#     recurring bottleneck we kept re-tuning: 600->450->350->300->150)
#   - a requirement that recent energy trend be non-negative (this was us
#     directly shaping what counts as "deserving" to reproduce -- if an
#     organism spends stale energy on offspring that then starve, that IS
#     selection working, not a bug to gate around)
# If reproduction now happens "too fast," that's information about the energy
# economy (a frozen-physics knob above), not a reason to re-add a gate here.
# Mutation, split into two regimes (see evolution.mutate):
#   - MICRO: a small gaussian drift on EVERY heritable continuous parameter
#     (weights + per-neuron threshold/decay/noise), applied on every birth.
#     Tiny per generation, but accumulates under selection -- a smooth walk.
#   - STRUCTURAL: a discrete topology change (add/remove edge, add node) at a
#     low per-birth rate, so innovations are occasional, not every-generation.
# To freeze evolution entirely (ablation), set BOTH to 0.0.
MICRO_MUTATION_SCALE = 0.02      # std of the always-on per-birth parameter drift
STRUCTURAL_MUTATION_RATE = 0.03  # per-birth chance of one discrete topology change
MAX_POPULATION = 300
SPAWN_RADIUS = 6                 # child appears within this many cells of parent

# ---------------------------------------------------------------------------
# Foraging regime (opt-in via --forage). The default economy rewards "camping"
# (sit on a patch), which never exercises perception->action loops. This regime
# makes food deplete and relocate, and scatters offspring onto open ground, so
# survival requires sensing energy and moving toward it — the pressure under
# which a perception-action loop should be selected.
# ---------------------------------------------------------------------------
PATCH_RELOCATE_ON_DEPLETION = False  # depleted patches respawn elsewhere
PATCH_RELOCATE_RADIUS = 12           # >0: food "drifts" locally; 0: teleports anywhere
PATCH_DRIFT_STEP = 0.0               # per-step random-walk of each patch (cells)
DISPERSE_RANDOM = False              # offspring spawn anywhere, not near parent

# Dynamic energy sources ("blooms"): patches age out (or deplete), despawn, and
# a fresh one spawns elsewhere — food supply stays constant but locations keep
# turning over, so survival requires finding new food rather than camping. The
# curriculum ramps this pressure in gradually so navigators are progressively
# favoured instead of the naive population dying all at once.
FORAGE_ENABLED = False               # master switch for the bloom dynamics
PATCH_MAX_AGE = 250                  # steps a bloom lives before expiring (lower = harsher)
PATCH_RESPAWN_RADIUS = 0             # 0 = bloom reappears anywhere; >0 = within radius
FORAGE_CURRICULUM_STEPS = 3000       # ramp pressure over this many steps (0 = instant)

# ---------------------------------------------------------------------------
# Run / logging
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS = 5000
LOG_EVERY = 100                  # collect summary stats every N timesteps
SEED = 42
