//! Effective MuJoCo contact parameters of the object (mjData.contact) for
//! jaw and table contacts, plus normal forces and gripper actuator saturation
//! over the scripted Grasp, with object DR off, mass-only, and friction DR.
//! JSON lines: "contact" (one per distinct parameter set) and "forces" (one per
//! Grasp phase: mean/max jaw and table normal force, actuator force / limit).
use mujoco_runtime::pick_place::{ObjectRandomization, PickPlaceConfig, PickPlaceSim};
use mujoco_runtime::skills::{CONTROL_PERIOD, Criteria, ScriptedGrasp, Skill, SkillStatus};

fn main() -> robotics_core::Result<()> {
    let fixed = |m: f64, f: Option<f64>| ObjectRandomization {
        mass: Some([m, m]),
        friction: f.map(|f| [f, f]),
        floor_friction: None,
    };
    let settings: [(&str, ObjectRandomization); 6] = [
        ("dr_off", ObjectRandomization::default()),
        ("mass_only_0.05", fixed(0.05, None)),
        ("friction_1.2", fixed(0.03, Some(1.2))),
        ("friction_0.1", fixed(0.03, Some(0.1))),
        ("m0.02_friction_0.1", fixed(0.02, Some(0.1))),
        ("m0.2_friction_0.1", fixed(0.2, Some(0.1))),
    ];
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_000_000);
    let mut pp = PickPlaceSim::new(PickPlaceConfig::default())?;
    for (name, r) in settings {
        pp.set_object_randomization(r)?;
        pp.reset(seed)?;
        let mut seen = Vec::new();
        let mut record = |pp: &PickPlaceSim, when: &str| {
            for c in pp.object_contact_parameters() {
                if c.dist > 1e-4 {
                    continue;
                }
                let key = format!("{:?}", (c.partner, c.dim, c.friction, c.solref, c.solimp));
                if !seen.contains(&key) {
                    seen.push(key);
                    println!(
                        "{}",
                        serde_json::json!({"type": "contact", "setting": name, "when": when,
                            "object": pp.object_physics(), "contact": c})
                    );
                }
            }
        };
        record(&pp, "rest");
        let mut grasp = ScriptedGrasp::new(Criteria::default());
        grasp.begin(&pp)?;
        // phase -> (steps, sum fixed, sum moving, max fixed, max moving, sum table,
        //           sum |actuator|/limit, steps at >= 99% of the limit)
        let mut phases: Vec<(&'static str, [f64; 8])> = Vec::new();
        while grasp.status(&pp) == SkillStatus::Running {
            let c = grasp.act(&pp)?;
            pp.step(&c, CONTROL_PERIOD)?;
            record(&pp, "grasp");
            let f = pp.grip_forces()?;
            let limit = f.gripper_force_range[1]
                .abs()
                .max(f.gripper_force_range[0].abs());
            let ratio = f.gripper_actuator_force.abs() / limit;
            let phase = grasp.phase().unwrap_or("?");
            if phases.last().is_none_or(|(p, _)| *p != phase) {
                phases.push((phase, [0.; 8]));
            }
            let a = &mut phases.last_mut().unwrap().1;
            a[0] += 1.;
            a[1] += f.fixed_jaw_normal;
            a[2] += f.moving_jaw_normal;
            a[3] = a[3].max(f.fixed_jaw_normal);
            a[4] = a[4].max(f.moving_jaw_normal);
            a[5] += f.table_normal;
            a[6] += ratio;
            a[7] += f64::from(u8::from(ratio >= 0.99));
        }
        for (phase, a) in phases {
            let n = a[0];
            println!(
                "{}",
                serde_json::json!({"type": "forces", "setting": name, "phase": phase,
                    "steps": n, "object": pp.object_physics(),
                    "fixed_jaw_normal_mean_n": a[1] / n, "moving_jaw_normal_mean_n": a[2] / n,
                    "fixed_jaw_normal_max_n": a[3], "moving_jaw_normal_max_n": a[4],
                    "table_normal_mean_n": a[5] / n,
                    "gripper_force_over_limit_mean": a[6] / n,
                    "steps_at_force_limit": a[7],
                    "grasp_status": format!("{:?}", grasp.status(&pp))})
            );
        }
    }
    Ok(())
}
