//! Effective MuJoCo contact parameters of the object (mjData.contact) for
//! jaw and table contacts, with object DR off, mass-only, and friction DR.
//! Prints one JSON line per (setting, partner, distinct parameter set).
use mujoco_runtime::pick_place::{ObjectRandomization, PickPlaceConfig, PickPlaceSim};
use mujoco_runtime::skills::{CONTROL_PERIOD, Criteria, ScriptedGrasp, Skill, SkillStatus};

fn main() -> robotics_core::Result<()> {
    let settings: [(&str, ObjectRandomization); 4] = [
        ("dr_off", ObjectRandomization::default()),
        (
            "mass_only_0.05",
            ObjectRandomization {
                mass: Some([0.05, 0.05]),
                friction: None,
            },
        ),
        (
            "friction_1.2",
            ObjectRandomization {
                mass: None,
                friction: Some([1.2, 1.2]),
            },
        ),
        (
            "friction_0.1",
            ObjectRandomization {
                mass: None,
                friction: Some([0.1, 0.1]),
            },
        ),
    ];
    let mut pp = PickPlaceSim::new(PickPlaceConfig::default())?;
    for (name, r) in settings {
        pp.set_object_randomization(r)?;
        pp.reset(3_000_000)?;
        let mut seen = Vec::new();
        let mut record = |pp: &PickPlaceSim, when: &str| {
            for c in pp.object_contact_parameters() {
                if c.dist > 1e-4 {
                    continue;
                }
                let key = (c.partner, c.dim, c.friction, c.solref, c.solimp);
                if !seen.contains(&format!("{key:?}")) {
                    seen.push(format!("{key:?}"));
                    println!(
                        "{}",
                        serde_json::json!({"setting": name, "when": when,
                            "object": pp.object_physics(), "contact": c})
                    );
                }
            }
        };
        record(&pp, "rest");
        let mut grasp = ScriptedGrasp::new(Criteria::default());
        grasp.begin(&pp)?;
        while grasp.status(&pp) == SkillStatus::Running {
            let c = grasp.act(&pp)?;
            pp.step(&c, CONTROL_PERIOD)?;
            record(&pp, "grasp");
        }
    }
    Ok(())
}
