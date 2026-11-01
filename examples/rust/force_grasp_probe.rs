//! Checks of the compliant-object force-grasp environment
//! (docs/FORCE_CONTROL.md §3–4). JSON lines on stdout.
//!
//! Usage: force_grasp_probe MODE [FIRST_SEED] [EPISODES]
//!   calibrate  requested vs. measured fixed-jaw gap (nominal object)
//!   static     step-and-hold squeeze: face force vs. spring law k·d,
//!              penetration vs. object mass and stiffness
//!   control    fixed close command vs. force PI on a goal-force grid
use mujoco_runtime::force_grasp::{CompliantObject, ForceGraspRanges, ForceGraspSim, ForcePi};
use mujoco_runtime::skills::{CONTROL_PERIOD, GRIPPER_CLOSED};
use robotics_core::Result;
use serde_json::json;

const TOUCH: f64 = 0.05;

/// Reproducible uniform [0, 1) from a seed (goal-force sampling).
fn unit(seed: u64) -> f64 {
    let z = (seed ^ 0x90a1_f0ce).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (z >> 11) as f64 / (1_u64 << 53) as f64
}

/// Close at `rate` rad per period until both jaws touch; returns steps used.
fn close_until_pinched(env: &mut ForceGraspSim, rate: f64) -> Result<Option<usize>> {
    let mut cmd = env.reading()?.gripper_command;
    for k in 0..400 {
        cmd -= rate;
        env.step(cmd)?;
        let r = env.reading()?;
        if r.jaw_force[0] + r.rigid_force[0] > TOUCH && r.jaw_force[1] + r.rigid_force[1] > TOUCH {
            return Ok(Some(k));
        }
    }
    Ok(None)
}

fn calibrate(env: &mut ForceGraspSim, first: u64, episodes: u64) -> Result<()> {
    for (width, gap) in [0.022, 0.026, 0.03, 0.032, 0.035]
        .into_iter()
        .flat_map(|w| [0.003, 0.004, 0.006, 0.008].map(|g| (w, g)))
    {
        for seed in first..first + episodes {
            let o = CompliantObject {
                width,
                gap,
                stiffness: 5000.,
                travel: 0.003,
                ..CompliantObject::nominal()
            };
            env.reset(seed, Some(o))?;
            let pinched = close_until_pinched(env, 0.002)?.is_some();
            let r = env.reading()?;
            println!(
                "{}",
                json!({"type": "calibrate", "seed": seed, "width_mm": width * 1e3,
                    "gap_requested_mm": gap * 1e3,
                    "pinched": pinched,
                    "positioning_displacement_mm": env.positioning_displacement * 1e3,
                    "push_before_pinch_mm": r.object_displacement * 1e3})
            );
        }
    }
    Ok(())
}

fn squeeze(env: &mut ForceGraspSim, first: u64, episodes: u64) -> Result<()> {
    let offsets = [0.0, -0.002, -0.005, -0.01, -0.02, -0.04, -0.08, -0.16];
    for stiffness in [200., 1000., 5000.] {
        for mass in [0.02, 0.1] {
            for seed in first..first + episodes {
                let o = CompliantObject {
                    stiffness,
                    damping: 2.
                        * 0.7
                        * (stiffness * mujoco_runtime::force_grasp::PLATE_MASS).sqrt(),
                    mass,
                    ..CompliantObject::nominal()
                };
                env.reset(seed, Some(o))?;
                if close_until_pinched(env, 0.002)?.is_none() {
                    println!(
                        "{}",
                        json!({"type": "static", "seed": seed, "pinched": false})
                    );
                    continue;
                }
                let q_c = env.reading()?.gripper_q;
                for delta in offsets {
                    let mut acc = Vec::new();
                    for k in 0..45 {
                        env.step(q_c + delta)?;
                        if k >= 30 {
                            acc.push(env.reading()?);
                        }
                    }
                    let last = *acc.last().unwrap();
                    if last.jaw_force[0].max(last.jaw_force[1])
                        > 3. * last.spring_force[0].max(last.spring_force[1]) + 5.
                    {
                        println!(
                            "{}",
                            json!({"type": "anomaly", "seed": seed, "stiffness": stiffness,
                                "mass": mass, "delta_cmd": delta, "reading": last,
                                "quat": env.pp.cube_pose().1, "contacts": env.object_contacts(),
                                "robot_table": env.robot_table_contacts()})
                        );
                    }
                    let n = acc.len() as f64;
                    let mean = |f: &dyn Fn(&mujoco_runtime::force_grasp::ForceReading) -> f64| {
                        acc.iter().map(f).sum::<f64>() / n
                    };
                    println!(
                        "{}",
                        json!({"type": "static", "seed": seed, "stiffness": stiffness,
                            "mass": mass, "delta_cmd": delta,
                            "q_minus_contact": mean(&|r| r.gripper_q) - q_c,
                            "jaw_force_n": [mean(&|r| r.jaw_force[0]), mean(&|r| r.jaw_force[1])],
                            "rigid_force_n": [mean(&|r| r.rigid_force[0]), mean(&|r| r.rigid_force[1])],
                            "spring_force_n": [mean(&|r| r.spring_force[0]), mean(&|r| r.spring_force[1])],
                            "deformation_mm": [mean(&|r| r.deformation[0]) * 1e3,
                                mean(&|r| r.deformation[1]) * 1e3],
                            "penetration_mm": mean(&|r| r.penetration) * 1e3,
                            "bottomed": last.bottomed, "travel_mm": o.travel * 1e3,
                            "torque_nm": mean(&|r| r.actuator_torque),
                            "object_displacement_mm": acc.last().unwrap().object_displacement * 1e3})
                    );
                }
            }
        }
    }
    Ok(())
}

struct Episode {
    forces: Vec<[f64; 2]>,
    peak: f64,
    displacement: f64,
    rigid: f64,
    saturated: usize,
    invalid: usize,
}

fn rollout(
    env: &mut ForceGraspSim,
    steps: usize,
    mut policy: impl FnMut(&mujoco_runtime::force_grasp::ForceReading) -> f64,
) -> Result<Episode> {
    let mut e = Episode {
        forces: Vec::new(),
        peak: 0.,
        displacement: 0.,
        rigid: 0.,
        saturated: 0,
        invalid: 0,
    };
    for _ in 0..steps {
        let r = env.reading()?;
        env.step(policy(&r))?;
        let r = env.reading()?;
        e.forces.push(r.jaw_force);
        e.peak = e.peak.max(r.jaw_force[0].max(r.jaw_force[1]));
        e.rigid = e.rigid.max(r.rigid_force[0].max(r.rigid_force[1]));
        e.displacement = r.object_displacement;
        e.saturated += usize::from(r.actuator_torque.abs() >= 0.999 * r.torque_limit);
        e.invalid += usize::from(r.penetration > mujoco_runtime::force_grasp::PENETRATION_LIMIT);
    }
    Ok(e)
}

fn control(env: &mut ForceGraspSim, first: u64, episodes: u64, args: &[f64]) -> Result<()> {
    let steps = 150;
    let dq_max = 0.03;
    let [kp, ki, k_est, speed] = match args {
        [a, b, c] => [*a, *b, *c, 0.6],
        [a, b, c, d, ..] => [*a, *b, *c, *d],
        _ => [0.5, 2.0, 100., 0.6],
    };
    let ranges = *env.ranges();
    for seed in first..first + episodes {
        let o = ranges.sample(seed);
        // Goal within what the object can bear (paper: f_goal < f_max(kappa)).
        let goal_hi = (0.8 * o.max_spring_force()).min(8.0);
        let goal = 0.3 + (goal_hi - 0.3).max(0.) * unit(seed);
        let names: &[&str] = if args.is_empty() {
            &["fixed_close", "force_pi"]
        } else {
            &["force_pi"]
        };
        for &name in names {
            env.reset(seed, Some(o))?;
            let mut pi = ForcePi::new(kp, ki, k_est, speed, TOUCH);
            let e = rollout(env, steps, |r| match name {
                "fixed_close" => GRIPPER_CLOSED,
                _ => {
                    let delta = pi
                        .act(r.jaw_force, goal, CONTROL_PERIOD)
                        .clamp(-dq_max, dq_max);
                    r.gripper_q + delta
                }
            })?;
            let tail = &e.forces[steps - 30..];
            let mean_f = tail.iter().map(|f| 0.5 * (f[0] + f[1])).sum::<f64>() / 30.;
            let err = tail
                .iter()
                .map(|f| ((f[0] - goal).abs() + (f[1] - goal).abs()) / 2.)
                .sum::<f64>()
                / 30.;
            let r = env.reading()?;
            println!(
                "{}",
                json!({"type": "control", "controller": name, "seed": seed, "object": o,
                    "goal_n": goal, "final_force_n": mean_f, "final_abs_error_n": err,
                    "relative_error": err / goal, "peak_force_n": e.peak,
                    "peak_rigid_force_n": e.rigid, "object_displacement_mm": e.displacement * 1e3,
                    "final_deformation_mm": [r.deformation[0] * 1e3, r.deformation[1] * 1e3],
                    "bottomed_out": r.deformation.iter().any(|d| *d >= 0.98 * o.travel),
                    "saturated_steps": e.saturated, "gains": [kp, ki, k_est, speed],
                    "invalid_contact_steps": e.invalid})
            );
        }
    }
    Ok(())
}

/// Per-period readings while closing slowly (debugging the set-up).
fn trace(env: &mut ForceGraspSim, seed: u64, gap: f64) -> Result<()> {
    let o = CompliantObject {
        gap,
        ..CompliantObject::nominal()
    };
    env.reset(seed, Some(o))?;
    let mut cmd = env.reading()?.gripper_command;
    for k in 0..300 {
        cmd -= 0.002;
        env.step(cmd)?;
        let r = env.reading()?;
        println!(
            "{}",
            json!({"type": "trace", "k": k, "reading": r,
            "pinch": env.pp.gripper_pose().0, "contacts": env.object_contacts(),
            "robot_table": env.robot_table_contacts(), "jaw_bodies": env.jaw_body_ids(),
            "q": env.pp.sim.state().position, "cmd": env.pp.sim.applied_command().position})
        );
        if r.jaw_force[0] > 1.0 && r.jaw_force[1] > 1.0 {
            break;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("static");
    let first: u64 = args
        .get(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000_000);
    let episodes: u64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(3);
    let rest: Vec<f64> = args.iter().skip(3).filter_map(|v| v.parse().ok()).collect();
    let mut env = ForceGraspSim::new(ForceGraspRanges::default())?;
    match mode {
        "calibrate" => calibrate(&mut env, first, episodes),
        "static" => squeeze(&mut env, first, episodes),
        "control" => control(&mut env, first, episodes, &rest),
        "pitrace" => {
            let o = env.ranges().sample(first);
            env.reset(first, Some(o))?;
            let goal = rest.first().copied().unwrap_or(1.0);
            let mut pi = ForcePi::new(0.5, 2.0, 100., 0.6, TOUCH);
            for k in 0..150 {
                let r = env.reading()?;
                let d = pi.act(r.jaw_force, goal, CONTROL_PERIOD).clamp(-0.03, 0.03);
                env.step(r.gripper_q + d)?;
                println!(
                    "{}",
                    json!({"type": "pitrace", "k": k, "object": o, "delta": d,
                    "reading": env.reading()?})
                );
            }
            Ok(())
        }
        "trace" => trace(&mut env, first, episodes as f64 * 1e-3),
        other => Err(robotics_core::Error::Input(format!("unknown mode {other}"))),
    }
}
