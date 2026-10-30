use clap::Parser;
use mujoco_runtime::{Simulator, presets};
use robotics_core::{
    JointCommand,
    task::{ReachTask, Task},
    telemetry::{NullTelemetry, TelemetrySink},
};
use std::path::PathBuf;

#[cfg(feature = "viewer")]
#[path = "../../so101-cli/src/viewer.rs"]
mod viewer;

#[derive(Parser)]
#[command(about = "Headless MuJoCo Reach with shared Rust robot/task API")]
struct Args {
    #[arg(long,default_value="so101",value_parser=["so101","panda","ur5e"])]
    robot: String,
    #[arg(long,default_value="reach",value_parser=["reach"])]
    task: String,
    #[arg(long, default_value_t = 1000)]
    steps: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 30.0)]
    control_hz: f64,
    #[arg(long,default_value="none",value_parser=["none","rerun"])]
    telemetry: String,
    /// Write an RRD instead of spawning the Viewer (requires --telemetry rerun).
    #[arg(long)]
    record: Option<PathBuf>,
    #[arg(long)]
    viewer: bool,
    #[arg(long)]
    trajectory: Option<PathBuf>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if !args.control_hz.is_finite() || args.control_hz <= 0.0 {
        return Err("control-hz must be finite and positive".into());
    }
    if args.record.is_some() && args.telemetry != "rerun" {
        return Err("--record requires --telemetry rerun".into());
    }
    if args.viewer && args.trajectory.is_some() {
        return Err("trajectory recording is headless only".into());
    }
    let mut sim = Simulator::from_spec(presets::robot(&args.robot)?)?;
    if args.trajectory.is_some() && 1.0 / args.control_hz < sim.timestep() {
        return Err("recording period must be >= physics timestep".into());
    }
    let mut task = ReachTask::default();
    task.reset(&sim, args.seed)?;
    let command = JointCommand {
        position: sim.state().position,
    };
    if args.viewer {
        if args.telemetry != "none" {
            return Err("use MuJoCo Viewer and Rerun separately in this version".into());
        }
        #[cfg(feature = "viewer")]
        {
            let mut hold = Hold(command.clone());
            viewer::run(&mut sim, &mut hold, args.steps)?;
        }
        #[cfg(not(feature = "viewer"))]
        return Err("rebuild with --features viewer to use --viewer".into());
    } else {
        let mut telemetry: Box<dyn TelemetrySink> = match args.telemetry.as_str() {
            "none" => Box::new(NullTelemetry),
            _ => {
                #[cfg(feature = "rerun")]
                {
                    Box::new(match &args.record {
                        Some(path) => telemetry_rerun::RerunTelemetry::save(path)?,
                        None => telemetry_rerun::RerunTelemetry::spawn()?,
                    })
                }
                #[cfg(not(feature = "rerun"))]
                return Err("rebuild with --features rerun to use --telemetry rerun".into());
            }
        };
        use robotics_core::trajectory::{NullRecorder, TrajectoryRecorder};
        let mut recorder: Box<dyn TrajectoryRecorder> = match &args.trajectory {
            Some(path) => Box::new(trajectory_io::CsvRecorder::new(path)),
            None => Box::new(NullRecorder),
        };
        recorder.begin(&sim.trajectory_metadata(1.0 / args.control_hz, args.seed)?)?;
        telemetry.begin_episode(1)?;
        for _ in 0..args.steps {
            let state = sim.step_control(&command, 1.0 / args.control_hz)?;
            recorder.record(&sim.trajectory_sample(&command)?)?;
            task.advance();
            telemetry.begin_step(sim.time(), task.episode_steps())?;
            for ((name, position), target) in sim
                .actuator_names()
                .iter()
                .zip(&state.position)
                .zip(&command.position)
            {
                telemetry.scalar(&format!("robot/joints/{name}"), *position)?;
                telemetry.scalar(&format!("robot/joint_targets/{name}"), *target)?;
            }
            telemetry.scalar("metrics/reward", task.reward(&sim)?)?;
            telemetry.scalar("metrics/distance", task.distance(&sim)?)?;
            telemetry.scalar("metrics/success", f64::from(task.terminated(&sim)?))?;
            telemetry.point3(
                "robot/end_effector",
                sim.end_effector(sim.default_end_effector()?)?,
            )?;
            telemetry.point3("task/target", task.target())?;
        }
        recorder.finish()?;
        telemetry.flush()?;
    }
    println!(
        "robot={} task={} joints={} time={:.6} reward={:.6} success={}",
        args.robot,
        args.task,
        sim.state().position.len(),
        sim.time(),
        task.reward(&sim)?,
        task.terminated(&sim)?
    );
    Ok(())
}
#[cfg(feature = "viewer")]
struct Hold(JointCommand);
#[cfg(feature = "viewer")]
impl robotics_core::controller::Controller for Hold {
    fn command(&mut self, _: &robotics_core::JointState, _: f64) -> JointCommand {
        self.0.clone()
    }
}
