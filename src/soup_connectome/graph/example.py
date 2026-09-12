from soup_connectome.config import SimulationConfig
from soup_connectome.graph.format import ConnectomeGraph, Edge, GraphBlock, NeuronRecord


def example_graph() -> ConnectomeGraph:
    """Return the deterministic fixture graph used by the default test suite.

    The topology and integer weights are design-fixture values, not biological
    measurements. Two source blocks make the resident/streamed contract
    observable without external data.
    """

    neurons = tuple(NeuronRecord(external_id=index) for index in range(4))
    blocks = (
        GraphBlock(
            source_start=0,
            rows=(
                (Edge(target=1, weight=32767, delay=1), Edge(target=2, weight=-12000, delay=1)),
                (Edge(target=3, weight=32767, delay=1),),
            ),
        ),
        GraphBlock(source_start=2, rows=((), ())),
    )
    return ConnectomeGraph(n_neurons=4, neurons=neurons, blocks=blocks)


def example_simulation_config() -> SimulationConfig:
    """Return intentionally small fixed-point fixture parameters."""

    return SimulationConfig(
        threshold=20000,
        reset=0,
        decay_shifts=(2,),
        refractory_steps=2,
    )
