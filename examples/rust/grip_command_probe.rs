//! Jaw normal force versus the gripper *position command* on the unchanged
//! SO-101 model (docs/FORCE_CONTROL.md §2). The scripted Grasp runs until the
//! Close phase starts; the arm command is then frozen and the gripper is
//! closed slowly until both jaws touch the cube (contact angle `q_c`). From
//! there the command steps through offsets `δ = q_cmd − q_c`, each held for
//! `HOLD` control periods, and the mean of the last third is reported:
//! gripper angle, actuator torque against its range, per-jaw normal force,
//! deepest jaw penetration and cube displacement.
//! JSON lines: one `contact` row per episode, one `level` row per offset.
//! Usage: grip_command_probe [FIRST_SEED] [EPISODES]
use mujoco_runtime::pick_place::{GRIPPER, ObjectRandomization, PickPlaceConfig, PickPlaceSim};
use mujoco_runtime::skills::{CONTROL_PERIOD, Criteria, ScriptedGrasp, Skill};
use robotics_core::Result;

const HOLD: usize = 30;
const OFFSETS: [f64; 14] = [
    0.004, 0.002, 0.0, -0.0005, -0.001, -0.002, -0.003, -0.004, -0.006, -0.01, -0.02, -0.05, -0.1,
    -0.3,
];
const TOUCH_N: f64 = 0.05;

struct Sample {
    q: f64,
    torque: f64,
    fixed: f64,
    moving: f64,
    table: f64,
    penetration: f64,
}

fn sample(pp: &PickPlaceSim) -> Result<Sample> {
    let f = pp.grip_forces()?;
    let penetration = pp
        .contact_details()
        .into_iter()
        .filter(|c| c.partner.ends_with("jaw"))
        .map(|c| -c.dist)
        .fold(0., f64::max);
    Ok(Sample {
        q: pp.sim.state().position[GRIPPER],
        torque: f.gripper_actuator_force,
        fixed: f.fixed_jaw_normal,
        moving: f.moving_jaw_normal,
        table: f.table_normal,
        penetration,
    })
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let first: u64 = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_000_000);
    let episodes: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(5);
    let mut pp = PickPlaceSim::new(PickPlaceConfig::default())?;
    for scale in [1.0, 0.5] {
        let mut p = pp.sim.parameters().clone();
        p.actuator_force_scale[GRIPPER] = scale;
        pp.sim.set_parameter_center(p)?;
        for mass in [0.02, 0.1] {
            pp.set_object_randomization(ObjectRandomization {
                mass: Some([mass, mass]),
                ..ObjectRandomization::default()
            })?;
            for seed in first..first + episodes {
                pp.reset(seed)?;
                let mut g = ScriptedGrasp::new(Criteria::default());
                g.begin(&pp)?;
                let mut command = g.act(&pp)?;
                while g.phase() != Some("close") {
                    pp.step(&command, CONTROL_PERIOD)?;
                    command = g.act(&pp)?;
                }
                let cube0 = pp.cube_pose().0;
                // Close slowly (2 mrad per control period) until both jaws touch.
                let mut cmd = pp.sim.state().position[GRIPPER];
                let mut touched = None;
                while cmd > -0.17 {
                    cmd -= 0.002;
                    command.position[GRIPPER] = cmd;
                    pp.step(&command, CONTROL_PERIOD)?;
                    let s = sample(&pp)?;
                    if s.fixed > TOUCH_N && s.moving > TOUCH_N {
                        touched = Some(s);
                        break;
                    }
                }
                let Some(touch) = touched else {
                    println!(
                        "{}",
                        serde_json::json!({"type": "contact", "seed": seed, "mass": mass,
                            "gripper_force_scale": scale, "pinched": false})
                    );
                    continue;
                };
                let q_c = touch.q;
                let push = (pp.cube_pose().0[0] - cube0[0]).hypot(pp.cube_pose().0[1] - cube0[1]);
                println!(
                    "{}",
                    serde_json::json!({"type": "contact", "seed": seed, "mass": mass,
                        "gripper_force_scale": scale, "pinched": true, "q_contact": q_c,
                        "command_at_contact": cmd, "cube_push_before_pinch_mm": push * 1e3})
                );
                for delta in OFFSETS {
                    let q_cmd = (q_c + delta).max(-0.17453);
                    command.position[GRIPPER] = q_cmd;
                    let mut acc = Vec::new();
                    let mut peak: f64 = 0.;
                    for k in 0..HOLD {
                        pp.step(&command, CONTROL_PERIOD)?;
                        let s = sample(&pp)?;
                        peak = peak.max(s.fixed.max(s.moving));
                        if k >= HOLD * 2 / 3 {
                            acc.push(s);
                        }
                    }
                    let n = acc.len() as f64;
                    let mean = |f: fn(&Sample) -> f64| acc.iter().map(f).sum::<f64>() / n;
                    let range = pp.grip_forces()?.gripper_force_range;
                    let torque = mean(|s| s.torque);
                    let cube = pp.cube_pose().0;
                    println!(
                        "{}",
                        serde_json::json!({"type": "level", "seed": seed, "mass": mass,
                            "gripper_force_scale": scale, "delta_cmd": delta, "q_cmd": q_cmd,
                            "q": mean(|s| s.q), "q_minus_contact": mean(|s| s.q) - q_c,
                            "torque_nm": torque, "torque_limit_nm": range[1],
                            "saturated": torque.abs() >= 0.999 * range[1].abs().min(range[0].abs()),
                            "fixed_normal_n": mean(|s| s.fixed),
                            "moving_normal_n": mean(|s| s.moving),
                            "table_normal_n": mean(|s| s.table), "peak_normal_n": peak,
                            "penetration_mm": mean(|s| s.penetration) * 1e3,
                            "cube_shift_mm": (cube[0] - cube0[0]).hypot(cube[1] - cube0[1]) * 1e3})
                    );
                }
            }
        }
    }
    Ok(())
}
