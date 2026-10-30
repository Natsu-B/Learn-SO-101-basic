//! Diagnostic: print gripper frame, jaw geometry and reachable pinch poses.
use mujoco_runtime::JointCommand;
use mujoco_runtime::pick_place::{IkTarget, PickPlaceConfig, PickPlaceSim};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pp = PickPlaceSim::new(PickPlaceConfig::default())?;
    pp.reset(0)?;
    println!("bounds {:?}", pp.sim.action_bounds());
    println!("approach_axis_local {:?}", pp.approach_axis_local);
    println!("jaw_local tip/mid/moving {:?}", pp.jaw_local);
    let s = pp.state();
    println!("home q {:?}", s.joint_position);
    println!(
        "home grip {:?} axis {:?}",
        s.gripper_position, s.gripper_axis
    );
    println!("cube {:?} target {:?}", s.cube_position, s.target_position);
    // Settle 1 s with home command; cube must remain at rest.
    let home = JointCommand {
        position: s.joint_position.clone(),
    };
    for _ in 0..30 {
        pp.step(&home, 1. / 30.)?;
    }
    let s = pp.state();
    println!(
        "after 1s cube {:?} v {:?} contacts {:?}",
        s.cube_position, s.cube_linear_velocity, s.contacts
    );
    for &g in &[-0.17, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5] {
        let mut q = s.joint_position.clone();
        q[5] = g;
        let cmd = JointCommand { position: q };
        for _ in 0..30 {
            pp.step(&cmd, 1. / 30.)?;
        }
        let j = pp.jaw_points_local()?;
        println!(
            "gripper cmd {g} q {:.3}: fixed tip {:?} moving tip {:?}",
            pp.sim.state().position[5],
            j[0],
            j[2]
        );
    }
    for &(x, y, z) in &[
        (0.15, 0.0, 0.0125),
        (0.25, 0.0, 0.0125),
        (0.2, 0.15, 0.0125),
        (0.2, 0.0, 0.08),
    ] {
        let (q, e) = pp.solve_ik(&s.joint_position, &IkTarget::top_down([x, y, z], None), 300)?;
        let (p, a) = pp.gripper_fk(&q)?;
        println!(
            "ik ({x},{y},{z}) err {e:.5} q {:?} p {:?} axis {:?}",
            q.iter()
                .map(|v| (v * 1000.).round() / 1000.)
                .collect::<Vec<_>>(),
            p,
            a
        );
    }
    Ok(())
}
