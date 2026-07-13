# SPECIES — V0 Working Prototype

A minimal simulation where sparse neural graphs live in a 2D world, spend energy
to exist and compute, harvest energy from patches, and replicate on surplus.
No backprop, no training loop — just a graph, an energy economy, local Hebbian
learning, and evolving topology.

## The actual research goal

The long-term target is **not** sparsity (that was a proxy). It is to watch for
**perception-action loops** — sensor input causally driving adaptive movement —
and eventually for distinct sub-regions of the graph to become attributable to
specific functions (the cortex / attention-head analogy). Everything below is
instrumented toward observing that.

## Run

```bash
cd species
python main.py                       # default (camping) regime, saves plots + stats
python main.py --timesteps 10000     # longer run
python main.py --forage              # foraging regime (drifting food) — experimental
python main.py --seed 7 --no-plots   # different seed, headless
```

## Files

| file | role |
|------|------|
| `config.py` | all constants (the energy economy lives here) |
| `organism.py` | Neuron/Synapse/Organism, threshold propagation, Hebbian update |
| `environment.py` | patch field, directional sensors, harvesting, movement |
| `evolution.py` | replication + structural mutation (add/remove edge, add node, perturb) |
| `simulation.py` | per-timestep orchestration and population management |
| `analysis.py` | **perception-action instrumentation** (see below) |
| `logger.py` | periodic summary stats → CSV |
| `visualizer.py` | world scatter + history plots (matplotlib, headless-safe) |
| `main.py` | CLI entry point |

## Where this deviates from the brief

The brief's raw constants do not self-bootstrap — random-walking founders can't
navigate to sparse patches and starve before reproducing, and a single patch
(cap 10) can't supply the replication threshold (20). Two changes were needed to
get a living, evolving population:

1. **Rebalanced economy** (`config.py`): denser/richer renewable patches, longer
   starting runway, wider spawn dispersal so offspring can colonize neighbouring
   patches before navigation has evolved.
2. **Senescence** (`SENESCENCE_SCALE`): metabolic cost grows with age. This is an
   addition beyond the brief's starvation-only death. Without it, successful
   "campers" become immortal, the population freezes at a subsistence pool, and
   evolution stalls (max generation flatlines). Set `SENESCENCE_SCALE = 0.0` to
   recover the original starvation-only behaviour.

## Perception-action instrumentation (`analysis.py`)

Three complementary probes, surfaced in the periodic log and `history.png`:

1. **Behavioural — `sensorimotor_alignment`** (the headline). Cosine similarity
   between the direction of sensed energy and the direction the organism
   actually moved. ~0 = blind wandering; > 0 = moving toward energy (a working
   loop); < 0 = fleeing energy.
2. **Structural — `frac_with_loop`, `mean_sa_pairs`.** Fraction of organisms with
   a directed sensor→actuator path, and how many such pairs. This is the wiring
   *substrate*; it can exist without being used.
3. **Causal — `matched_reflex_rate`, `mean_influence`.** Activate one sensor,
   propagate, read the actuators. `matched_reflex_rate` measures whether sensor
   *k* drives the *same-direction* actuator (N sensor → move north) — an
   adaptive reflex arc, and the first step toward attributing function to
   structure.

## Findings so far (seed 42)

**The substrate is universal; the function is absent.** In the default regime:

- `frac_with_loop = 1.00` — *every* organism has sensor→actuator wiring. Random
  sparse graphs already contain the loop substrate.
- `sensorimotor_alignment ≈ 0.00` — movement is uncorrelated with sensed energy.
  Organisms wander blindly.
- `matched_reflex_rate ≈ 0.03` — a sensor almost never drives its matching
  actuator.

The loop exists but is **functionally dead**, because the default economy rewards
*camping* (sit on a static patch): you never need to sense or move to survive, so
nothing selects for tuning the loop. This shows up spatially too — the *fullest*
patches in `world.png` sit unoccupied; lineages stay local to where an ancestor
happened to land.

**Foraging pressure does not (yet) make the loop emerge — it causes extinction.**
The `--forage` regime makes food drift so tracking it should pay. But across
drift levels the population goes **extinct** rather than evolving navigation:
random networks can't forage, and structural mutation (~2.5%/birth) plus Hebbian
tuning are far too slow to discover a sensorimotor reflex before the drifting
food starves the population. This is a genuine result: **the perception-action
loop does not bootstrap for free** under these mechanisms.

Meta-point: sparsity was similarly a non-starter — sweeping `EDGE_COST_SCALE`
either barely moves density or drives extinction, for the same timescale reason.

## Suggested next steps (toward emergent loops)

- **Make the loop learnable faster.** The bottleneck is the rate of adaptive
  structural/weight change. Options: much stronger/faster plasticity on
  sensor→actuator paths, or a developmental bias that preferentially wires
  sensors toward actuators, giving selection real variance to grab.
- **Curriculum, not cliff.** Start from an established (camping) population, then
  *ramp* drift up slowly so navigators are progressively favoured instead of the
  whole naive population dying at once.
- **Then look for modularity.** Once alignment is reliably > 0, use
  `influence_matrix` to ask whether specific subgraphs are responsible for
  specific sensor→actuator reflexes — the first evidence of functional regions.
