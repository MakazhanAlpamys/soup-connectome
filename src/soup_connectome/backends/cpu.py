from soup_connectome.config import Residency, SimulationConfig
from soup_connectome.graph.format import ConnectomeGraph
from soup_connectome.sim.runtime import SimulationResult, run_graph


class CPUBackend:
    name = "cpu"
    available = True

    def run(
        self,
        graph: ConnectomeGraph,
        config: SimulationConfig,
        *,
        timesteps: int,
        initial_potentials: tuple[int, ...] | None = None,
        residency: Residency | str = Residency.resident,
    ) -> SimulationResult:
        return run_graph(
            graph,
            config,
            timesteps=timesteps,
            initial_potentials=initial_potentials,
            residency=residency,
        )
