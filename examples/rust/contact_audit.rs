//! Physics audit of the SO-101 grasp (docs/PHYSICS_AUDIT.md).
//!
//! JSON lines on stdout:
//! - "model": solver options, servo parameters, jaw/cube/table contact geoms.
//! - "params": each distinct resolved contact parameter set per setting.
//! - "trace": one row per physics step (5 ms) of the scripted Grasp from its
//!   Close phase on: gripper state and actuator force, cube pose/velocity and
//!   finite-difference acceleration, gripperframe axes, and every cube contact
//!   (position, normal and force on the cube, penetration).
//! - "equivalence": stepping 5 ms at a time reproduces control-rate stepping.
//!
//! Usage: contact_audit [FIRST_SEED] [EPISODES]   (default 3000000 4)
use mujoco_runtime::pick_place::{
    ContactModel, ContactOverrides, ObjectRandomization, PickPlaceConfig, PickPlaceSim,
};
use mujoco_runtime::skills::{CONTROL_PERIOD, Criteria, ScriptedGrasp, Skill, SkillStatus};
use robotics_core::{JointCommand, Result};
use serde_json::json;

struct Setting {
    name: &'static str,
    model: ContactModel,
    overrides: ContactOverrides,
    mass: Option<f64>,
    friction: Option<f64>,
    floor: Option<f64>,
}
const fn setting(
    name: &'static str,
    model: ContactModel,
    mass: Option<f64>,
    friction: Option<f64>,
    floor: Option<f64>,
) -> Setting {
    Setting {
        name,
        model,
        overrides: ContactOverrides {
            jaw_condim: None,
            jaw_solref: None,
            floor_condim: None,
            floor_solref: None,
        },
        mass,
        friction,
        floor,
    }
}

fn sim(s: &Setting) -> Result<PickPlaceSim> {
    let mut pp = PickPlaceSim::new(PickPlaceConfig {
        contact_model: s.model,
        contact_overrides: s.overrides,
        ..PickPlaceConfig::default()
    })?;
    pp.set_object_randomization(ObjectRandomization {
        mass: s.mass.map(|m| [m, m]),
        friction: s.friction.map(|f| [f, f]),
        floor_friction: s.floor.map(|f| [f, f]),
    })?;
    Ok(pp)
}

fn model_row(pp: &PickPlaceSim) -> Result<serde_json::Value> {
    let names = pp.sim.actuator_names();
    let params = pp.sim.physical_parameters()?;
    let bounds = pp.sim.action_bounds();
    let actuators: Vec<_> = names
        .iter()
        .zip(&params)
        .zip(&bounds)
        .map(|((n, p), b)| {
            json!({"name": n, "damping": p[0], "frictionloss": p[1], "kp": p[2],
                   "bias_kp": p[3], "bias_kv": p[4], "forcerange": [p[5], p[6]],
                   "ctrlrange": b})
        })
        .collect();
    let geom = |g| pp.geom_info(g);
    let jaws = pp
        .jaw_geoms()
        .iter()
        .map(|&g| geom(g))
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"type": "model", "solver": pp.solver_options(), "actuators": actuators,
        "jaw_geoms": jaws, "cube_geom": geom(pp.cube_geom())?, "table_geom": geom(pp.table_geom())?,
        "object": pp.object_physics(), "jaw_points_site": pp.jaw_points_local()?,
        "grasp_offset_site": pp.grasp_offset}),
    )
}

fn tilt(q: [f64; 4]) -> f64 {
    (1. - 2. * (q[1] * q[1] + q[2] * q[2]))
        .clamp(-1., 1.)
        .acos()
}

/// Scripted Grasp to its end. With `trace`, the physics is advanced one 5 ms
/// step at a time (same schedule as control-rate stepping) and logged from Close.
fn grasp(
    pp: &mut PickPlaceSim,
    seed: u64,
    name: &str,
    trace: bool,
) -> Result<(SkillStatus, [f64; 3])> {
    pp.reset(seed)?;
    let mut g = ScriptedGrasp::new(Criteria::default());
    g.begin(pp)?;
    let dt = pp.sim.timestep();
    let mut k = 0u64;
    let mut seen: Vec<String> = Vec::new();
    let gripper = pp.sim.actuator_names().len() - 1;
    while g.status(pp) == SkillStatus::Running {
        let command: JointCommand = g.act(pp)?;
        let phase = g.phase().unwrap_or("");
        k += 1;
        let logging = trace && !matches!(phase, "approach" | "descend");
        if !trace {
            pp.step(&command, CONTROL_PERIOD)?;
            continue;
        }
        let deadline = k as f64 * CONTROL_PERIOD;
        while pp.sim.time() + dt * 1e-9 < deadline {
            let v0 = pp.cube_velocity();
            pp.sim.step_physics(&command)?;
            if !logging {
                continue;
            }
            let v1 = pp.cube_velocity();
            let (pos, quat) = pp.cube_pose();
            let (center, approach) = pp.gripper_pose();
            let closing = pp.closing_direction();
            let f = pp.grip_forces()?;
            let js = pp.sim.state();
            let contacts: Vec<_> = pp
                .contact_details()
                .into_iter()
                .map(|c| {
                    let key = format!(
                        "{}|{}|{}|{:?}|{:?}",
                        c.partner, c.dim, c.friction, c.solref, c.solimp
                    );
                    if !seen.contains(&key) {
                        println!(
                            "{}",
                            json!({"type": "params", "setting": name,
                            "partner": c.partner, "geom": c.geom, "dim": c.dim,
                            "friction": c.friction, "solref": c.solref, "solimp": c.solimp})
                        );
                        seen.push(key);
                    }
                    json!({"partner": c.partner, "geom": c.geom, "geom_id": c.geom_id,
                           "dist": c.dist, "pos": c.pos, "normal": c.normal,
                           "force": c.force, "frame_force": c.frame_force})
                })
                .collect();
            println!(
                "{}",
                json!({"type": "trace", "setting": name, "seed": seed, "control_step": k,
                    "t": pp.sim.time(), "phase": phase,
                    "gripper_q": js.position[gripper], "gripper_qd": js.velocity[gripper],
                    "gripper_command": command.position[gripper],
                    "gripper_force": f.gripper_actuator_force,
                    "cube_pos": pos, "cube_quat": quat, "cube_vel": v1,
                    "cube_acc": std::array::from_fn::<f64, 6, _>(|i| (v1[i] - v0[i]) / dt),
                    "tilt": tilt(quat), "grasp_center": center, "approach": approach,
                    "closing": closing, "mass": pp.object_physics().mass,
                    "contacts": contacts})
            );
        }
    }
    Ok((g.status(pp), pp.cube_pose().0))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let first: u64 = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_000_000);
    let episodes: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(4);
    use ContactModel::{Legacy, Pairwise};
    let mut settings = vec![
        setting("legacy_dr_off", Legacy, None, None, None),
        setting(
            "legacy_m0.035_mu1.15",
            Legacy,
            Some(0.035),
            Some(1.15),
            None,
        ),
        setting("legacy_m0.02_mu0.1", Legacy, Some(0.02), Some(0.1), None),
        setting("legacy_m0.2_mu0.1", Legacy, Some(0.2), Some(0.1), None),
        setting("pairwise_dr_off", Pairwise, None, None, None),
        setting(
            "pairwise_m0.02_jaw0.1",
            Pairwise,
            Some(0.02),
            Some(0.1),
            None,
        ),
        setting(
            "pairwise_m0.02_jaw0.1_floor0.1",
            Pairwise,
            Some(0.02),
            Some(0.1),
            Some(0.1),
        ),
        setting("pairwise_m0.2_jaw0.1", Pairwise, Some(0.2), Some(0.1), None),
    ];
    let mut condim4 = setting(
        "pairwise_m0.02_jaw0.1_condim4",
        Pairwise,
        Some(0.02),
        Some(0.1),
        None,
    );
    condim4.overrides.jaw_condim = Some(4);
    settings.push(condim4);
    println!("{}", model_row(&sim(&settings[0])?)?);
    for s in &settings {
        let mut pp = sim(s)?;
        for seed in first..first + episodes {
            let (status, traced) = grasp(&mut pp, seed, s.name, true)?;
            let (plain_status, plain) = grasp(&mut pp, seed, s.name, false)?;
            println!(
                "{}",
                json!({"type": "equivalence", "setting": s.name, "seed": seed,
                    "status": format!("{status:?}"), "plain_status": format!("{plain_status:?}"),
                    "object": pp.object_physics(),
                    "cube_final_max_abs_diff": (0..3).map(|i| (traced[i] - plain[i]).abs())
                        .fold(0., f64::max)})
            );
        }
    }
    Ok(())
}
