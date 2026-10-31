"""Pairwise contact environment, low-dimensional residual Grasp and Place
variants (docs/PHYSICS_AUDIT.md)."""
import numpy as np
import pytest

from robot_sim import PickPlaceSimulator
from robot_sim.grasp_env import GoalGraspEnv, LowDimResidualGrasp, residual_episode
from robot_sim.grasp_learning import wrist_roll_policy
from robot_sim.pick_place_pipeline import run_pick_place_episode


@pytest.fixture
def pairwise():
    simulator = PickPlaceSimulator(contact_model="pairwise")
    try:
        yield simulator
    finally:
        simulator.close()


def test_pairwise_friction_is_split_and_reported(pairwise):
    pairwise.set_object_randomization(mass=(0.02, 0.02), friction=(0.1, 0.1))
    pairwise.reset(3)
    p = pairwise.object_physics()
    assert (p["contact_model"], p["jaw_friction"], p["floor_friction"], p["priority"]) == (
        "pairwise", 0.1, 1.2, 0)
    pairwise.set_object_randomization(friction=(0.3, 0.3), floor_friction=(0.2, 0.2))
    pairwise.reset(3)
    p = pairwise.object_physics()
    assert (p["jaw_friction"], p["floor_friction"]) == (0.3, 0.2)
    table = [c for c in pairwise.contact_details() if c["partner"] == "table"]
    assert table and all(c["dist"] <= 1e-4 for c in table)


def test_legacy_rejects_floor_friction_and_unknown_models():
    sim = PickPlaceSimulator()
    try:
        with pytest.raises(Exception):
            sim.set_object_randomization(floor_friction=(0.5, 0.5))
        assert sim.object_physics()["contact_model"] == "legacy"
    finally:
        sim.close()
    with pytest.raises(ValueError):
        PickPlaceSimulator(contact_model="bogus")


def _final_state(sim, seed, options):
    run_pick_place_episode(seed, transport="scripted", sim=sim, **options)
    return np.asarray(sim.state()["cube_pos"])


def test_low_dim_zero_action_equals_fixed_wrist_offset(pairwise):
    fixed = wrist_roll_policy(-0.1, end_at="base")
    config = dict(fixed.config, action_mode="wrist", base_wrist=-0.1, wrist_scale=0.1)

    class Zero:
        learned = True

        def predict(self, obs, deterministic=True):
            return np.zeros(1, np.float32), None

    a = _final_state(pairwise, 7, dict(grasp="learned", grasp_model=fixed,
                                        grasp_config=fixed.config))
    b = _final_state(pairwise, 7, dict(grasp="learned", grasp_model=Zero(),
                                        grasp_config=config))
    # The fixed baseline maps its offset through a float32 action (-0.1 ->
    # -0.10000000149), so only rounding-level differences are allowed.
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_low_dim_actions_are_validated_and_lift_adjustment_acts(pairwise):
    from robot_sim.grasp_env import approach_to_pre_close
    bounds = np.asarray(pairwise.action_bounds())
    criteria = pairwise.criteria()
    config = dict(history=0, horizon=75, arm_scale=0.06, gripper_scale=0.35, end_at="base",
                  action_mode="wrist_lift", base_wrist=0.0, wrist_scale=0.1,
                  lift_scale=0.01, speed_scale=0.5)
    heights = []
    for speed in (-1.0, 1.0):
        pairwise.reset(2)
        assert approach_to_pre_close(pairwise)
        episode = residual_episode(pairwise, config, criteria=criteria, bounds=bounds)
        assert isinstance(episode, LowDimResidualGrasp)
        with pytest.raises(ValueError):
            episode.step(np.zeros(6))
        for _ in range(30):
            episode.step(np.array([0.0, 0.0, speed]))
        heights.append(episode.state["cube_pos"][2])
    # A faster lift raises the cube further in the same number of steps.
    assert heights[1] > heights[0] + 0.01
    pairwise.start_skill("transport")
    with pytest.raises(Exception):
        pairwise.skill_adjust([0.0, 1.0])
    pairwise.start_skill("grasp")
    with pytest.raises(Exception):
        pairwise.skill_adjust([0.5, 1.0])


def test_env_action_space_follows_mode():
    env = GoalGraspEnv(action_mode="wrist", base_wrist=-0.1, contact_model="pairwise",
                       floor_friction="same", friction_sampling="log",
                       mass_range=(0.02, 0.1), friction_range=(0.05, 1.5))
    try:
        assert env.action_space.shape == (1,)
        obs, _ = env.reset(seed=4)
        p = env.sim.object_physics()
        assert p["jaw_friction"] == p["floor_friction"]
        obs, reward, terminated, truncated, info = env.step(np.zeros(1, np.float32))
        assert np.isfinite(reward) and obs.shape == env.observation_space.shape
    finally:
        env.close()
    with pytest.raises(ValueError):
        GoalGraspEnv(floor_friction="same")  # needs pairwise


def test_place_variants_run_and_settle_is_logged_separately(pairwise):
    for place in ("place_gentle", "place_level_contact"):
        row = run_pick_place_episode(5, transport="scripted", sim=pairwise, place=place)
        assert row["success"], row["skills"]
        assert row["skills"][-1]["variant"] == place
    row = run_pick_place_episode(5, transport="scripted", sim=pairwise,
                                 settle_before_place=True)
    assert row["success"] and row["settle"]["success"]
    assert [r["name"] for r in row["skills"]] == ["grasp", "transport", "place"]
    with pytest.raises(ValueError):
        run_pick_place_episode(5, transport="scripted", sim=pairwise, place="drop")
