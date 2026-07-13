"""Rendering: a spatial view of the world and time-series of the run stats.

Two entry points:
  - render_world(sim): scatter of organisms over the energy patches (single frame)
  - plot_history(rows): the logged summary series over time

Both save PNGs by default so the sim can run headless (no display needed).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; overridden by caller for live windows
import matplotlib.pyplot as plt


def render_world(sim, path: str | None = "world.png", show: bool = False):
    fig, ax = plt.subplots(figsize=(7, 7))
    env = sim.env

    ax.scatter(
        env.patch_pos[:, 0],
        env.patch_pos[:, 1],
        s=env.patch_energy * 8 + 1,
        c="tab:green",
        alpha=0.5,
        label="energy patches",
    )
    if sim.population:
        xs = [o.position[0] for o in sim.population]
        ys = [o.position[1] for o in sim.population]
        gens = [o.generation for o in sim.population]
        sc = ax.scatter(xs, ys, s=12, c=gens, cmap="plasma", label="organisms")
        fig.colorbar(sc, ax=ax, label="generation")

    ax.set_xlim(0, sim.grid_size)
    ax.set_ylim(0, sim.grid_size)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # +y = South, matching the sensor convention
    ax.set_title(f"ADNS world  t={sim.timestep}  pop={len(sim.population)}")
    ax.legend(loc="upper right")

    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_history(rows: list[dict], path: str | None = "history.png", show: bool = False):
    if not rows:
        return
    t = [r["timestep"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, [r["population"] for r in rows], color="tab:blue")
    axes[0, 0].set_title("Population")

    # Perception-action panel: the headline observables for emergent loops.
    axes[0, 1].plot(t, [r.get("sensorimotor_alignment", 0.0) for r in rows],
                    color="tab:red", label="sensorimotor alignment")
    axes[0, 1].plot(t, [r.get("matched_reflex_rate", 0.0) for r in rows],
                    color="tab:orange", label="matched reflex rate")
    axes[0, 1].plot(t, [r.get("frac_with_loop", 0.0) for r in rows],
                    color="tab:green", label="frac. with S->A path")
    axes[0, 1].axhline(0.0, color="gray", lw=0.6)
    axes[0, 1].set_title("Perception-action loop")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, [r["mean_nodes"] for r in rows], label="nodes")
    axes[1, 0].plot(t, [r["mean_edges"] for r in rows], label="edges")
    axes[1, 0].set_title("Mean network size")
    axes[1, 0].legend()

    axes[1, 1].plot(t, [r["max_generation"] for r in rows], color="tab:purple")
    axes[1, 1].set_title("Max generation")

    for ax in axes.flat:
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
