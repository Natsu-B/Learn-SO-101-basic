//! Scripted (IK) Grasp -> Transport -> Place over many seeds; prints JSON lines
//! and a summary. No learned model is used here (baseline).
use mujoco_runtime::pick_place::{PickPlaceConfig, PickPlaceSim};
use mujoco_runtime::skills::{Criteria, run_episode, scripted_skills};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let episodes: u64 = args.get(1).map(|v| v.parse()).transpose()?.unwrap_or(20);
    let first: u64 = args.get(2).map(|v| v.parse()).transpose()?.unwrap_or(0);
    let criteria = Criteria::default();
    let mut env = PickPlaceSim::new(PickPlaceConfig::default())?;
    let mut counts = [0usize; 3];
    let mut ok = 0;
    for seed in first..first + episodes {
        let mut skills = scripted_skills(&criteria);
        let trace = std::env::var_os("PICK_TRACE").is_some();
        let mut step = 0;
        let report = run_episode(&mut env, &mut skills, seed, |e, name| {
            step += 1;
            if trace && step % 5 == 0 {
                let s = e.state();
                let r = |v: &[f64]| {
                    v.iter()
                        .map(|x| (x * 1000.).round() / 1000.)
                        .collect::<Vec<_>>()
                };
                eprintln!(
                    "{step} {name} grip {:?} axis {:?} cube {:?} q {:?} gq {:.3} {:?}",
                    r(&s.gripper_position),
                    r(&s.gripper_axis),
                    r(&s.cube_position),
                    r(&s.joint_position),
                    s.joint_position[5],
                    s.contacts
                );
            }
        })?;
        for (i, s) in report.skills.iter().enumerate() {
            counts[i] += usize::from(s.success);
        }
        ok += usize::from(report.success);
        println!("{}", serde_json::to_string(&report)?);
    }
    eprintln!(
        "episodes {episodes}: grasp {} transport {} place {} overall {}",
        counts[0], counts[1], counts[2], ok
    );
    Ok(())
}
