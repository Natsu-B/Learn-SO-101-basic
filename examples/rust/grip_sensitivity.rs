//! Sensitivity of the scripted (IK) pick-and-place to the gripper force limit,
//! jaw-pad friction and table friction, in the pairwise contact environment
//! (docs/PHYSICS_AUDIT.md §1). The gripper's forcerange is scaled through the
//! simulator's parameter centre (2.94 N·m = STS3215 stall torque at scale 1).
//! JSON lines, one per condition: grasp / hold / success counts and the mean
//! jaw normal force while lifted.
//! Usage: grip_sensitivity [FIRST_SEED] [EPISODES]
use mujoco_runtime::pick_place::{
    ContactModel, GRIPPER, ObjectRandomization, PickPlaceConfig, PickPlaceSim,
};
use mujoco_runtime::skills::{Criteria, predicates, run_episode, scripted_skills};
use robotics_core::Result;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let first: u64 = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_000_000);
    let episodes: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(30);
    let mut pp = PickPlaceSim::new(PickPlaceConfig {
        contact_model: ContactModel::Pairwise,
        ..PickPlaceConfig::default()
    })?;
    let criteria = Criteria::default();
    for scale in [1.0, 0.5] {
        let mut p = pp.sim.parameters().clone();
        p.actuator_force_scale[GRIPPER] = scale;
        pp.sim.set_parameter_center(p)?;
        for mass in [0.02, 0.1] {
            for jaw in [1.0, 0.3, 0.1] {
                for floor in [1.2, 0.1] {
                    pp.set_object_randomization(ObjectRandomization {
                        mass: Some([mass, mass]),
                        friction: Some([jaw, jaw]),
                        floor_friction: Some([floor, floor]),
                    })?;
                    let (mut grasp, mut hold, mut ok) = (0, 0, 0);
                    let (mut force, mut samples) = (0.0, 0usize);
                    for seed in first..first + episodes {
                        let r = run_episode(
                            &mut pp,
                            &mut scripted_skills(&criteria),
                            seed,
                            |e, name| {
                                if name == "transport"
                                    && predicates::lifted(e, &criteria)
                                    && let Ok(f) = e.grip_forces()
                                {
                                    force += 0.5 * (f.fixed_jaw_normal + f.moving_jaw_normal);
                                    samples += 1;
                                }
                            },
                        )?;
                        grasp += usize::from(r.skills.first().is_some_and(|s| s.success));
                        hold += usize::from(r.skills.get(1).is_some_and(|s| s.success));
                        ok += usize::from(r.success);
                    }
                    println!(
                        "{}",
                        serde_json::json!({"gripper_force_scale": scale,
                            "gripper_torque_limit_nm": 2.94 * scale, "mass": mass,
                            "jaw_friction": jaw, "floor_friction": floor, "episodes": episodes,
                            "grasp": grasp, "hold": hold, "success": ok,
                            "transport_jaw_normal_mean_n": force / samples.max(1) as f64})
                    );
                }
            }
        }
    }
    Ok(())
}
