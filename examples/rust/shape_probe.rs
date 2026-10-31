//! Scripted pick-and-place success per object shape (feasibility probe).
//! Args: shape (cube|sphere), episodes, first seed. Summary on stdout.
use mujoco_runtime::pick_place::{ObjectShape, PickPlaceConfig, PickPlaceSim};
use mujoco_runtime::skills::{Criteria, run_episode, scripted_skills};

fn main() -> robotics_core::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let shape = match args.get(1).map(String::as_str).unwrap_or("sphere") {
        "cube" => ObjectShape::Cube,
        _ => ObjectShape::Sphere,
    };
    let n: u64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(30);
    let first: u64 = args
        .get(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_000_000);
    let mut pp = PickPlaceSim::new(PickPlaceConfig {
        shape,
        ..Default::default()
    })?;
    let c = Criteria::default();
    let mut failures = std::collections::BTreeMap::<String, usize>::new();
    let mut success = [0usize; 4];
    for seed in first..first + n {
        let r = run_episode(&mut pp, &mut scripted_skills(&c), seed, |_, _| {})?;
        for (i, s) in r.skills.iter().enumerate() {
            success[i] += usize::from(s.success);
            if let Some(reason) = &s.reason {
                *failures.entry(format!("{}: {reason}", s.name)).or_default() += 1;
            }
        }
        success[3] += usize::from(r.success);
    }
    println!(
        "{shape:?}: episodes {n}, grasp {}, transport {}, place {}, all {}, failures {failures:?}",
        success[0], success[1], success[2], success[3]
    );
    Ok(())
}
