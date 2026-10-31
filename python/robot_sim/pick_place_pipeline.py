"""Scripted Grasp -> learned (or scripted) Transport -> scripted Place.

No reset between skills, no fallback, and all physical decisions use Rust.
"""
import numpy as np

from ._native import CONTROL_PERIOD, GRIPPER_CLOSED, PickPlaceSimulator
from .transport_env import transport_command, transport_observation


def chain_succeeded(reports):
    """The chain from the last Grasp attempt (after any fallback) is
    Grasp [-> Stabilize] -> Transport -> Place, and every step succeeded."""
    names = [r["name"] for r in reports]
    start = len(names) - 1 - names[::-1].index("grasp") if "grasp" in names else 0
    chain = reports[start:]
    return ([r["name"] for r in chain] in (["grasp", "transport", "place"],
                                           ["grasp", "stabilize", "transport", "place"])
            and all(r["success"] for r in chain))


def skill_end_state(sim, name):
    """Physical hand-off state after a skill (true state, for logs only)."""
    from .grasp_env import cube_tilt

    state = sim.state()
    half = sim.criteria()["cube_half"]
    velocity = np.asarray(state["cube_vel"], np.float64)
    offset = np.subtract(state["cube_pos"], state["grasp_center"])
    return dict(name=name, height_m=float(state["cube_pos"][2] - half),
                speed_mps=float(np.linalg.norm(velocity[:3])),
                angular_speed_rps=float(np.linalg.norm(velocity[3:])),
                tilt_deg=float(np.degrees(cube_tilt(state["cube_quat"]))),
                gripper_rad=float(state["gripper"]), cube_in_jaws_m=offset.tolist(),
                pinched=bool(state["contacts"]["pinched"]),
                table=bool(state["contacts"]["table"]))


def run_learned_grasp(sim, model, config, *, on_step=None):
    """Scripted approach (IK) to the pre-close state, then the residual policy.

    Uses the same ResidualGrasp core as GoalGraspEnv, without reset or retry.
    Returns a skill report; ``learned`` is True only if a learned policy acted
    (a model with ``learned = False``, such as a fixed residual, reports False).
    """
    from .grasp_env import ResidualGrasp, approach_to_pre_close

    approach_steps = 0

    def count(state, name, learned):
        nonlocal approach_steps
        approach_steps += 1
        if on_step is not None:
            on_step(state, name, learned)

    if not approach_to_pre_close(sim, count):
        report = sim.skill_report()
        return dict(name="grasp", learned=False, success=False, status="failure",
                    steps=approach_steps, reason=f"approach: {report['reason']}",
                    approach_steps=approach_steps, max_slip_m=0.0)
    episode = ResidualGrasp(sim, history=config["history"], horizon=config["horizon"],
                            arm_scale=config["arm_scale"], gripper_scale=config["gripper_scale"],
                            criteria=sim.criteria(),
                            bounds=np.asarray(sim.action_bounds(), np.float64),
                            end_at=config.get("end_at") or (
                                "base" if config.get("reward") == "v3" else "success"))
    while not episode.done:
        action, _ = model.predict(episode.obs, deterministic=True)
        episode.step(action)
        if on_step is not None:
            on_step(episode.state, "grasp", True)
    t = episode.tracker
    ok = episode.success
    return dict(name="grasp", learned=bool(getattr(model, "learned", True)), success=ok,
                status="success" if ok else "failure",
                steps=approach_steps + episode.step_count,
                reason=None if ok else ("dropped" if t.dropped else "not held"),
                approach_steps=approach_steps, max_slip_m=t.max_slip)


def run_pick_place_episode(seed, *, model=None, transport="learned", sim=None,
                           max_transport_steps=150, max_delta=0.05, rate_limit=0.1,
                           on_step=None, grasp="scripted", grasp_model=None, grasp_config=None,
                           grasp_fallback=False, stabilize=False):
    """Run one native seed and return an EpisodeReport-compatible dictionary.

    ``model`` is a loaded SAC policy with ``predict(obs, deterministic=True)``.
    Pass the checkpoint's horizon/max_delta/rate_limit for learned inference.
    Scripted transport uses Rust's own step limit. ``on_step(state, name,
    learned)`` receives a fresh snapshot after every actual control step, for
    all three skills. A supplied simulator remains open; otherwise it is owned
    here and closed even if inference or the callback raises.

    ``grasp="learned"`` runs the residual Grasp policy (``grasp_model`` with its
    checkpoint ``grasp_config``). With ``grasp_fallback`` a failed learned
    Grasp is followed by a fresh scripted Grasp from the resulting state (no
    reset); both attempts are reported and the fallback is ``learned=False``.
    ``stabilize`` inserts the scripted Stabilize skill (IK: carry height, wait
    until still) between a successful Grasp and Transport; it is reported as
    its own skill and counts toward overall success. ``stabilize="best_effort"``
    proceeds after the time limit if the cube is still lifted.

    ``place`` selects the scripted Place variant: "place", "place_level"
    (rotate the hand about the closing axis so the cube is level at release),
    "place_contact" (lower until the cube touches the table, then open) or
    "place_level_contact", "place_gentle" (open only to 0.25 rad before
    retreating) or "place_gentle_contact". ``settle_before_place`` runs the best-effort IK
    Stabilize (stop at carry height, wait until still) between Transport and
    Place, reported as its own "settle" skill that does not decide success.

    ``skill_end_states`` records height, speed, tilt, gripper angle and cube
    offset in the jaws after every attempted skill (true state, logs only).

    As in Rust, only attempted skills appear in the report. ``dropped`` records
    Rust's dropped predicate during Transport: the initial table state and the
    intentional release during Place are not accidental drops.
    """
    if transport not in ("learned", "scripted"):
        raise ValueError("transport must be learned or scripted")
    if transport == "learned" and model is None:
        raise ValueError("learned transport requires a model")
    if grasp not in ("learned", "scripted"):
        raise ValueError("grasp must be learned or scripted")
    if place not in ("place", "place_level", "place_contact", "place_level_contact",
                     "place_gentle", "place_gentle_contact"):
        raise ValueError("unknown place variant")
    if grasp == "learned" and (grasp_model is None or grasp_config is None):
        raise ValueError("learned grasp requires grasp_model and grasp_config")
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

        end_states = []
        settle_report = None
        if grasp == "learned":
            reports.append(run_learned_grasp(sim, grasp_model, grasp_config, on_step=on_step))
            end_states.append(skill_end_state(sim, "grasp"))
            if not reports[-1]["success"] and grasp_fallback:
                fallback = scripted("grasp")
                fallback["fallback"] = True
                reports.append(fallback)
                end_states.append(skill_end_state(sim, "grasp"))
        else:
            reports.append(scripted("grasp"))
            end_states.append(skill_end_state(sim, "grasp"))
        if reports[-1]["success"] and stabilize:
            reports.append(scripted("stabilize_best_effort" if stabilize == "best_effort"
                                    else "stabilize"))
            reports[-1]["name"] = "stabilize"
            end_states.append(skill_end_state(sim, "stabilize"))
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
            end_states.append(skill_end_state(sim, "transport"))
            if reports[-1]["success"] and settle_before_place:
                settle = scripted("stabilize_best_effort")
                settle["name"] = "settle"
                end_states.append(skill_end_state(sim, "settle"))
                # A failed settle (drop) ends the chain; a successful one is
                # logged separately so the Grasp/Transport/Place chain is unchanged.
                if settle["success"]:
                    settle_report = settle
                else:
                    reports.append(settle)
            if reports[-1]["success"]:
                report = scripted(place)
                if place != "place":
                    report["name"] = "place"
                    report["variant"] = place
                reports.append(report)
                end_states.append(skill_end_state(sim, "place"))

        final = sim.state()
        placed = sim.predicates()["placed"]
        return dict(seed=int(seed), cube_initial=initial["cube_pos"], target=final["target"],
                    cube_final=final["cube_pos"], final_xy_error=float(np.linalg.norm(
                        np.asarray(final["cube_pos"])[:2] - np.asarray(final["target"])[:2])),
                    success=bool(chain_succeeded(reports) and placed),
                    skills=reports, control_steps=sum(r["steps"] for r in reports),
                    dropped=bool(dropped), skill_end_states=end_states, settle=settle_report)
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
