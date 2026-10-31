//! Jaw-contact penetration and normal force at the end of a successful IK
//! Grasp, versus object mass and gripper force limit (pairwise contact
//! environment, scene friction). MuJoCo's soft contacts are acceleration-based,
//! so at a fixed squeeze force a lighter object sinks deeper into the pads.
//! JSON lines, one per (scale, mass).
//! Usage: penetration_probe [FIRST_SEED] [EPISODES] [JAW_SOLREF0 JAW_SOLREF1]
use mujoco_runtime::pick_place::{
    ContactModel, ContactOverrides, GRIPPER, ObjectRandomization, PickPlaceConfig, PickPlaceSim,
};
use mujoco_runtime::skills::{CONTROL_PERIOD, Criteria, ScriptedGrasp, Skill, SkillStatus};
use robotics_core::Result;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let first: u64 = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_000_000);
    let episodes: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(10);
    let solref: Vec<f64> = args.filter_map(|v| v.parse().ok()).collect();
    let jaw_solref = (solref.len() == 2).then(|| [solref[0], solref[1]]);
    let mut pp = PickPlaceSim::new(PickPlaceConfig {
        contact_model: ContactModel::Pairwise,
        contact_overrides: ContactOverrides {
            jaw_solref,
            ..ContactOverrides::default()
        },
        ..PickPlaceConfig::default()
    })?;
    for scale in [1.0, 0.5] {
        let mut p = pp.sim.parameters().clone();
        p.actuator_force_scale[GRIPPER] = scale;
        pp.sim.set_parameter_center(p)?;
        for mass in [0.01, 0.02, 0.05, 0.1, 0.2] {
            pp.set_object_randomization(ObjectRandomization {
                mass: Some([mass, mass]),
                ..ObjectRandomization::default()
            })?;
            let (mut pen, mut normal, mut count, mut n) = (Vec::new(), 0.0, 0.0, 0.0);
            for seed in first..first + episodes {
                pp.reset(seed)?;
                let mut g = ScriptedGrasp::new(Criteria::default());
                g.begin(&pp)?;
                while g.status(&pp) == SkillStatus::Running {
                    let c = g.act(&pp)?;
                    pp.step(&c, CONTROL_PERIOD)?;
                }
                if g.status(&pp) != SkillStatus::Success {
                    continue;
                }
                let jaws: Vec<_> = pp
                    .contact_details()
                    .into_iter()
                    .filter(|c| c.partner.ends_with("jaw"))
                    .collect();
                pen.push(jaws.iter().map(|c| -c.dist).fold(0., f64::max));
                normal += jaws.iter().map(|c| c.frame_force[0]).sum::<f64>() / 2.;
                count += jaws.len() as f64;
                n += 1.;
            }
            let mean = pen.iter().sum::<f64>() / n;
            println!(
                "{}",
                serde_json::json!({"gripper_force_scale": scale, "mass": mass,
                    "jaw_solref": jaw_solref, "grasped": n,
                    "episodes": episodes, "max_penetration_mean_mm": mean * 1e3,
                    "max_penetration_max_mm": pen.iter().cloned().fold(0., f64::max) * 1e3,
                    "jaw_normal_per_side_n": normal / n, "jaw_contacts": count / n})
            );
        }
    }
    Ok(())
}
