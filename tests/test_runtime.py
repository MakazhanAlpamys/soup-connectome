from soup_connectome.graph.example import example_graph, example_simulation_config
from soup_connectome.sim.runtime import run_graph


def test_streamed_reference_matches_resident_bit_for_bit() -> None:
    graph = example_graph()
    config = example_simulation_config()
    resident = run_graph(
        graph,
        config,
        timesteps=4,
        initial_potentials=(32767, 0, 0, 0),
        residency="resident",
    )
    streamed = run_graph(
        graph,
        config,
        timesteps=4,
        initial_potentials=(32767, 0, 0, 0),
        residency="streamed",
    )

    assert streamed == resident
