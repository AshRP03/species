"""Summary statistics collected periodically over the run.

Tracks the observables the concept brief cares about: population size, network
size and density (is sparsity emerging?), firing rate, and the deepest lineage.
Stored as a list of dict rows so it is trivial to dump to CSV or plot.
"""

from __future__ import annotations

import csv
from statistics import mean

import config
from analysis import causal_summary, structural_summary


def _safe_mean(values) -> float:
    values = list(values)
    return float(mean(values)) if values else 0.0


class Logger:
    def __init__(self):
        self.rows: list[dict] = []

    def snapshot(self, sim) -> dict:
        pop = sim.population
        n = len(pop)
        densities = [
            len(o.synapses) / len(o.neurons) for o in pop if o.neurons
        ]
        firing = [
            _safe_mean(nu.state for nu in o.neurons.values()) for o in pop
        ]
        struct = structural_summary(pop)
        causal = causal_summary(pop, rng=getattr(sim, "rng", None))
        row = {
            "timestep": sim.timestep,
            "population": n,
            "mean_energy": _safe_mean(o.energy for o in pop),
            "mean_nodes": _safe_mean(len(o.neurons) for o in pop),
            "mean_edges": _safe_mean(len(o.synapses) for o in pop),
            "mean_density": _safe_mean(densities),
            "mean_firing_rate": _safe_mean(firing),
            "max_generation": max((o.generation for o in pop), default=0),
            # --- perception-action observables (the headline metrics) ---
            # Interval-averaged alignment (per-step value is too noisy to read).
            "sensorimotor_alignment": sim.alignment_window() if hasattr(sim, "alignment_window") else 0.0,
            "frac_with_loop": struct["frac_with_loop"],
            "mean_sa_pairs": struct["mean_sa_pairs"],
            "mean_influence": causal["mean_influence"],
            "matched_reflex_rate": causal["matched_reflex_rate"],
        }
        self.rows.append(row)
        return row

    def maybe_snapshot(self, sim) -> dict | None:
        if sim.timestep % config.LOG_EVERY == 0:
            return self.snapshot(sim)
        return None

    def to_csv(self, path: str) -> None:
        if not self.rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

    @staticmethod
    def format_row(row: dict) -> str:
        return (
            f"t={row['timestep']:>6}  "
            f"pop={row['population']:>4}  "
            f"gen={row['max_generation']:>3}  "
            f"align={row['sensorimotor_alignment']:>+5.2f}  "
            f"loop={row['frac_with_loop']:>4.2f}  "
            f"reflex={row['matched_reflex_rate']:>4.2f}  "
            f"E={row['mean_energy']:>5.1f}  "
            f"edges={row['mean_edges']:>5.1f}"
        )
