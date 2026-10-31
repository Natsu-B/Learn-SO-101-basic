"""Object mass/friction randomization through the native binding."""
import pytest

from robot_sim import PickPlaceSimulator


@pytest.fixture
def sim():
    simulator = PickPlaceSimulator()
    try:
        yield simulator
    finally:
        simulator.close()


def test_default_is_scene_physics_and_state_hides_it(sim):
    sim.reset(1)
    p = sim.object_physics()
    assert (p["mass"], p["friction"], p["priority"]) == (0.03, 1.2, 0)
    assert p["mass_range"] is None and p["friction_range"] is None
    # Policy-facing snapshots never carry mass or friction.
    assert not {"mass", "friction", "object_physics"} & set(sim.state())


def test_ranges_apply_at_reset_and_are_seeded(sim):
    sim.reset(4)
    placement = (sim.state()["cube_pos"], sim.state()["target"])
    sim.set_object_randomization(mass=(0.02, 0.05), friction=(0.8, 1.5))
    assert sim.object_physics()["mass"] == 0.03  # unchanged until reset
    sim.reset(4)
    p = sim.object_physics()
    assert 0.02 <= p["mass"] <= 0.05 and 0.8 <= p["friction"] <= 1.5
    assert p["priority"] >= 2
    assert (sim.state()["cube_pos"], sim.state()["target"]) == placement
    sim.reset(5)
    sim.reset(4)
    assert sim.object_physics() == p
    sim.set_object_randomization(mass=(0.04, 0.04))
    sim.reset(4)
    q = sim.object_physics()
    assert q["mass"] == 0.04 and q["friction"] == 1.2 and q["priority"] == 0


@pytest.mark.parametrize("kwargs", [dict(mass=(0.05, 0.02)), dict(mass=(0.0, 0.1)),
                                    dict(friction=(0.8, 10.0))])
def test_invalid_ranges_raise(sim, kwargs):
    with pytest.raises(ValueError):
        sim.set_object_randomization(**kwargs)
