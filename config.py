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

# ---------------------------------------------------------------------------
# Organism
# ---------------------------------------------------------------------------
INIT_POPULATION = 50
INIT_NODES = 40                  # interneurons + sensors + actuators at spawn
INIT_EDGE_PROB = 0.05            # sparse random graph
SENSOR_COUNT = 4                 # N / S / E / W energy proximity
ACTUATOR_COUNT = 4               # N / S / E / W movement drive
INIT_ENERGY = 20.0               # energy a founder organism starts with (survival runway)
INIT_WEIGHT_SCALE = 0.5          # std of initial synapse weights

# ---------------------------------------------------------------------------
# Energy costs / gains
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

# ---------------------------------------------------------------------------
# Local learning (Hebbian + energy gate)
# ---------------------------------------------------------------------------
LEARNING_RATE = 0.01
WEIGHT_PRUNE_THRESHOLD = 0.02    # synapses weaker than this are removed
WEIGHT_CLIP = 5.0                # keep weights bounded for numerical sanity

# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------
REPLICATION_THRESHOLD = 30.0     # energy needed to spawn a child
MUTATION_RATE = 0.1              # probability a structural mutation fires on birth
WEIGHT_PERTURB_SCALE = 0.1       # std of gaussian noise on the perturb mutation
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

# ---------------------------------------------------------------------------
# Run / logging
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS = 5000
LOG_EVERY = 100                  # collect summary stats every N timesteps
SEED = 42
