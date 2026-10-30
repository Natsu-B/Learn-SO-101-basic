use mujoco_runtime::{Simulator, presets};
use robotics_core::{
    JointCommand,
    task::{ReachTask, Task},
};

pub fn run(robot: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut sim = Simulator::from_spec(presets::robot(robot)?)?;
    let mut task = ReachTask::default();
    task.reset(&sim, 42)?;
    let action = JointCommand {
        position: sim.state().position,
    };
    for _ in 0..1000 {
        sim.step_control(&action, 1.0 / 30.0)?;
    }
    println!(
        "{robot}: time={} reward={} target={:?}",
        sim.time(),
        task.reward(&sim)?,
        task.target()
    );
    Ok(())
}
