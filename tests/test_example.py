from soup_connectome.graph.example import example_graph, example_simulation_config
from soup_connectome.sim.runtime import run_graph


def test_example_graph_has_expected_shape() -> None:
    graph = example_graph()

    assert graph.n_neurons == 4
    assert graph.edge_count == 3
    assert graph.min_delay == 1


def test_example_has_golden_spike_train() -> None:
    result = run_graph(
        example_graph(),
        example_simulation_config(),
        timesteps=4,
        initial_potentials=(32767, 0, 0, 0),
        residency="resident",
    )

    assert result.spikes == (
        (True, False, False, False),
        (False, True, False, False),
        (False, False, False, True),
        (False, False, False, False),
    )
