"""Scripted Grasp -> learned (or scripted) Transport -> scripted Place.

No reset between skills, no fallback, and all physical decisions use Rust.
"""
import numpy as np

from ._native import CONTROL_PERIOD, GRIPPER_CLOSED, PickPlaceSimulator
from .transport_env import transport_command, transport_observation


def run_pick_place_episode(seed, *, model=None, transport="learned", sim=None,
                           max_transport_steps=150, max_delta=0.05, rate_limit=0.1,
                           on_step=None):
    """Run one native seed and return an EpisodeReport-compatible dictionary.

    ``model`` is a loaded SAC policy with ``predict(obs, deterministic=True)``.
    Pass the checkpoint's horizon/max_delta/rate_limit for learned inference.
    Scripted transport uses Rust's own step limit. ``on_step(state, name,
    learned)`` receives a fresh snapshot after every actual control step, for
    all three skills. A supplied simulator remains open; otherwise it is owned
    here and closed even if inference or the callback raises.

    As in Rust, only attempted skills appear in the report. ``dropped`` records
    Rust's dropped predicate during Transport: the initial table state and the
    intentional release during Place are not accidental drops.
    """
    if transport not in ("learned", "scripted"):
        raise ValueError("transport must be learned or scripted")
    if transport == "learned" and model is None:
        raise ValueError("learned transport requires a model")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    if (isinstance(max_transport_steps, bool)
            or not isinstance(max_transport_steps, (int, np.integer)) or max_transport_steps < 1):
        raise ValueError("max_transport_steps must be a positive integer")
    for name, value in (("max_delta", max_delta), ("rate_limit", rate_limit)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    owns_sim = sim is None
    if owns_sim:
        sim = PickPlaceSimulator()
    try:
        sim.reset(int(seed))
        initial = sim.state()
        reports = []
        dropped = False

        def scripted(name):
            nonlocal dropped
            # Use Rust's bulk runner when no intermediate snapshots are needed.
            if on_step is None and name != "transport":
                return sim.run_skill(name)
            sim.start_skill(name)
            status = sim.skill_report()["status"]
            while status == "running":
                status = sim.skill_step()
                if name == "transport":
                    dropped |= sim.predicates()["dropped"]
                if on_step is not None:
                    on_step(sim.state(), name, False)
            return sim.skill_report()

        reports.append(scripted("grasp"))
        if reports[-1]["success"]:
            if transport == "scripted":
                reports.append(scripted("transport"))
            else:
                state = sim.state()
                criteria = sim.criteria()
                carry_z = criteria["cube_half"] + criteria["carry_height"]
                bounds = np.asarray(sim.action_bounds(), np.float64)
                previous = np.asarray(state["qpos"], np.float64).copy()
                previous[5] = GRIPPER_CLOSED
                steps = 0
                predicates = sim.predicates()
                while not predicates["transported"] and not predicates["dropped"]:
                    if steps >= max_transport_steps:
                        break
                    obs = transport_observation(state, carry_z)
                    action, _ = model.predict(obs, deterministic=True)
                    previous = transport_command(state, action, previous, bounds,
                                                 max_delta=max_delta, rate_limit=rate_limit)
                    state = sim.step(previous.tolist(), CONTROL_PERIOD)
                    steps += 1
                    predicates = sim.predicates()
                    dropped |= predicates["dropped"]
                    if on_step is not None:
                        on_step(state, "transport", True)
                dropped |= predicates["dropped"]
                success = predicates["transported"]
                reports.append(dict(name="transport", learned=True, success=success,
                                    status="success" if success else "failure", steps=steps,
                                    reason=None if success else
                                    ("dropped" if predicates["dropped"] else "timeout")))
            if reports[-1]["success"]:
                reports.append(scripted("place"))

        final = sim.state()
        placed = sim.predicates()["placed"]
        return dict(seed=int(seed), cube_initial=initial["cube_pos"], target=final["target"],
                    cube_final=final["cube_pos"], final_xy_error=float(np.linalg.norm(
                        np.asarray(final["cube_pos"])[:2] - np.asarray(final["target"])[:2])),
                    success=bool(len(reports) == 3 and all(r["success"] for r in reports)
                                 and placed),
                    skills=reports, control_steps=sum(r["steps"] for r in reports),
                    dropped=bool(dropped))
    finally:
        if owns_sim:
            sim.close()


def summarize_pick_place(outcomes, *, transport):
    """Rates use all episodes; attempts expose skills skipped after a failure."""
    if not outcomes:
        raise ValueError("at least one episode is required")
    errors = np.asarray([row["final_xy_error"] for row in outcomes])
    skills = {}
    for name in ("grasp", "transport", "place"):
        reports = [r for row in outcomes for r in row["skills"] if r["name"] == name]
        successes = sum(r["success"] for r in reports)
        skills[name] = dict(learned=name == "transport" and transport == "learned",
                            attempts=len(reports), successes=successes,
                            success_rate=successes / len(outcomes),
                            success_rate_when_attempted=successes / len(reports) if reports else None)
    return dict(type="summary", episodes=len(outcomes), seed=outcomes[0]["seed"],
                transport=transport, success_rate=sum(r["success"] for r in outcomes) / len(outcomes),
                skills=skills, drop_count=sum(r["dropped"] for r in outcomes),
                final_xy_error_m=dict(mean=float(errors.mean()),
                                      p95=float(np.percentile(errors, 95)), max=float(errors.max())),
                learned_skills=["transport"] if transport == "learned" else [])
