"""SPECIES V0 entry point.

Run the simulation end-to-end, print periodic stats, and (optionally) save
plots of the world and the run history.

    python main.py                        # default run, saves PNGs + stats CSV
    python main.py --timesteps 2000       # shorter run
    python main.py --seed 7 --no-plots    # different seed, headless, no images
"""

from __future__ import annotations

import argparse

import config
from logger import Logger
from simulation import Simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPECIES V0 simulation")
    p.add_argument("--timesteps", type=int, default=config.DEFAULT_TIMESTEPS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--grid-size", type=int, default=config.GRID_SIZE)
    p.add_argument("--no-plots", action="store_true", help="skip PNG output")
    p.add_argument("--stats-csv", type=str, default="stats.csv")
    p.add_argument(
        "--forage",
        action="store_true",
        help="foraging regime: food depletes and relocates, offspring disperse "
        "randomly. Applies selection pressure for perception-action loops.",
    )
    return p.parse_args()


def apply_foraging_regime() -> None:
    # Dynamic blooms: food supply stays constant but locations turn over, so
    # camping fails and tracking sensed energy pays off. A curriculum ramps the
    # pressure in gradually so navigators are progressively favoured rather than
    # the naive founder population dying all at once.
    config.FORAGE_ENABLED = True
    config.PATCH_MAX_AGE = 250          # bloom lifespan after the curriculum ramp
    config.PATCH_RESPAWN_RADIUS = 0     # blooms reappear anywhere -> real search
    config.FORAGE_CURRICULUM_STEPS = 3000
    config.SPAWN_RADIUS = 8             # moderate offspring dispersal
    config.SENESCENCE_SCALE = 0.0003


def main() -> None:
    args = parse_args()
    if args.forage:
        apply_foraging_regime()
    sim = Simulation(seed=args.seed, grid_size=args.grid_size)
    logger = Logger()

    print(f"SPECIES V0 — {args.timesteps} timesteps, seed={args.seed}, grid={args.grid_size}")
    print(f"Founders: {len(sim.population)} organisms\n")

    logger.snapshot(sim)  # t=0 baseline

    def on_step(s: Simulation) -> None:
        row = logger.maybe_snapshot(s)
        if row is not None:
            print(Logger.format_row(row))

    sim.run(args.timesteps, on_step=on_step)

    final = logger.snapshot(sim)
    print("\nFinal state:")
    print(Logger.format_row(final))

    if not sim.population:
        print("\n** Population went extinct. **")
    else:
        print(f"\nSurvived to t={sim.timestep} with {len(sim.population)} organisms.")

    logger.to_csv(args.stats_csv)
    print(f"Stats written to {args.stats_csv}")

    if not args.no_plots:
        from visualizer import plot_history, render_world

        render_world(sim, path="world.png")
        plot_history(logger.rows, path="history.png")
        print("Plots written to world.png and history.png")


if __name__ == "__main__":
    main()
