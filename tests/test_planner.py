from soup_connectome.config import RuntimeAxes
from soup_connectome.graph.example import example_graph
from soup_connectome.sim.planner import plan_graph


def test_planner_is_cpu_only_and_marks_unknown_capacity() -> None:
    plan = plan_graph(
        example_graph(),
        RuntimeAxes(device="cpu", residency="resident", scope="full"),
        capacity_bytes=None,
    )

    assert plan.device == "cpu"
    assert plan.capacity_status == "not tested"
    assert plan.required_bytes > 0
