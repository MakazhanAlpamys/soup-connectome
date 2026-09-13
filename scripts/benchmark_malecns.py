from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from soup_connectome.backends.base import resolve_backend
from soup_connectome.config import SimulationConfig
from soup_connectome.graph.format import DiskGraphArtifact, load_artifact, open_artifact

DEFAULT_DATASET = Path("artifacts/male-cns-v1.0-annotated.scx")
DEFAULT_CONFIG = SimulationConfig(
    threshold=20000,
    reset=0,
    decay_shifts=(2,),
    refractory_steps=2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the local full-scale MaleCNS artifact.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", choices=("cpu", "webgpu", "cuda"), default="cpu")
    parser.add_argument("--residency", choices=("streamed", "resident"), default="streamed")
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--seed-neuron", type=int, default=0)
    parser.add_argument("--seed-potential", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps < 0:
        raise SystemExit("--timesteps must be non-negative")
    if not args.dataset.is_dir():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    graph = (
        open_artifact(args.dataset)
        if args.residency == "streamed"
        else load_artifact(args.dataset).graph
    )
    if not isinstance(graph, DiskGraphArtifact) and args.residency == "streamed":
        raise AssertionError("streamed loader did not return a disk artifact")
    if not 0 <= args.seed_neuron < graph.n_neurons:
        raise SystemExit("--seed-neuron is outside the graph")
    initial = [0] * graph.n_neurons
    initial[args.seed_neuron] = args.seed_potential
    started = time.perf_counter()
    result = resolve_backend(args.device).run(
        graph,
        DEFAULT_CONFIG,
        timesteps=args.timesteps,
        initial_potentials=tuple(initial),
        residency=args.residency,
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "evidence": "measured",
                "dataset": str(args.dataset),
                "device": args.device,
                "residency": args.residency,
                "timesteps": args.timesteps,
                "n_neurons": graph.n_neurons,
                "n_edges": graph.edge_count,
                "seed_neuron": args.seed_neuron,
                "seed_potential": args.seed_potential,
                "config": DEFAULT_CONFIG.model_dump(mode="json"),
                "wall_seconds": round(elapsed, 6),
                "timestep_spike_counts": [sum(step) for step in result.spikes],
                "spikes": result.spike_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
