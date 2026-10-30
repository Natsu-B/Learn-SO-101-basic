use mujoco_runtime::{Error, JointCommand, Simulator as CoreSimulator};
use robotics_core::{
    Result,
    task::{ReachTask, Task},
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{Mutex, mpsc},
    thread,
};
pub(crate) struct Session {
    pub(crate) sim: CoreSimulator,
    pub(crate) task: ReachTask,
    pub(crate) episode: u64,
    pub(crate) recording_period: Option<f64>,
    pub(crate) recorder: Option<Box<dyn robotics_core::trajectory::TrajectoryRecorder>>,
    pub(crate) telemetry: Option<Box<dyn robotics_core::telemetry::TelemetrySink>>,
}
impl Session {
    pub(crate) fn snapshot(&self) -> Result<robotics_core::task::StepResult> {
        let observation = self.sim.observed();
        Ok(robotics_core::task::StepResult {
            joint_state: observation.joint_state,
            end_effector_position: observation
                .end_effector_position
                .ok_or_else(|| Error::Model("missing default end effector".into()))?,
            target_position: self.task.target(),
            reward: self.task.reward(&self.sim)?,
            distance: self.task.distance(&self.sim)?,
            terminated: self.task.terminated(&self.sim)?,
            truncated: self.task.truncated(&self.sim)?,
            simulation_time: self.sim.time(),
            observation_time: observation.timestamp,
            episode_step: self.task.episode_steps(),
        })
    }

    pub(crate) fn check_recording_period(&self, period: f64) -> Result<()> {
        if self
            .recording_period
            .is_some_and(|p| (p - period).abs() > 1e-12)
        {
            return Err(Error::Input(
                "step period does not match trajectory metadata".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn finish_recording(&mut self) -> Result<()> {
        self.recording_period = None;
        if let Some(mut recorder) = self.recorder.take() {
            recorder.finish()?;
        }
        Ok(())
    }
    pub(crate) fn log(&mut self, command: &JointCommand) -> robotics_core::Result<()> {
        if let Some(recorder) = &mut self.recorder {
            recorder.record(&self.sim.trajectory_sample(command)?)?;
        }
        if let Some(sink) = &mut self.telemetry {
            sink.begin_step(self.sim.time(), self.task.episode_steps())?;
            if self.task.episode_steps() == 0 {
                let p = self.sim.parameters();
                for (field, values) in [
                    ("damping_scale", &p.joint_damping_scale),
                    ("frictionloss_scale", &p.frictionloss_scale),
                    ("motor_gain_scale", &p.actuator_gain_scale),
                    ("force_scale", &p.actuator_force_scale),
                    ("sensor_zero_offset_rad", &p.sensor_zero_offset_rad),
                ] {
                    for (name, value) in p.joint_names.iter().zip(values) {
                        sink.scalar(&format!("domain_randomization/{field}/{name}"), *value)?;
                    }
                }
                sink.scalar(
                    "domain_randomization/control_latency_seconds",
                    p.control_latency_seconds,
                )?;
                sink.scalar(
                    "domain_randomization/observation_latency_seconds",
                    p.observation_latency_seconds,
                )?;
            }
            let observed = self.sim.observed().joint_state.position;
            let applied = self.sim.applied_command().position.clone();
            for (i, ((name, value), target)) in self
                .sim
                .actuator_names()
                .iter()
                .zip(self.sim.state().position)
                .zip(&command.position)
                .enumerate()
            {
                sink.scalar(&format!("robot/joints/{name}"), value)?;
                sink.scalar(&format!("control/requested/{name}"), *target)?;
                sink.scalar(&format!("control/applied/{name}"), applied[i])?;
                sink.scalar(&format!("state/true/{name}"), value)?;
                sink.scalar(&format!("state/observed/{name}"), observed[i])?;
                sink.scalar(&format!("robot/joint_targets/{name}"), *target)?;
            }
            sink.scalar("metrics/reward", self.task.reward(&self.sim)?)?;
            sink.scalar("metrics/distance", self.task.distance(&self.sim)?)?;
            sink.scalar(
                "metrics/success",
                f64::from(self.task.terminated(&self.sim)?),
            )?;
            sink.point3(
                "robot/end_effector",
                self.sim.end_effector(self.sim.default_end_effector()?)?,
            )?;
            sink.point3("task/target", self.task.target())?;
        }
        Ok(())
    }
}

type Job = Box<dyn FnOnce(&mut Session) + Send>;
pub(crate) struct Worker {
    sender: Mutex<Option<mpsc::Sender<Job>>>,
    handle: Mutex<Option<thread::JoinHandle<()>>>,
}
fn stopped() -> Error {
    Error::Worker("simulator is poisoned or closed; create a new Simulator".into())
}
impl Worker {
    pub(crate) fn new(model_dir: Option<PathBuf>, robot: String) -> Result<Self> {
        let (sender, receiver) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("mujoco-worker".into())
            .spawn(move || {
                let initialized = catch_unwind(AssertUnwindSafe(|| {
                    let sim = match model_dir {
                        Some(dir) => CoreSimulator::from_model_dir(dir),
                        None => CoreSimulator::from_spec(mujoco_runtime::presets::robot(&robot)?),
                    }?;
                    Ok::<_, Error>(Session {
                        sim,
                        task: ReachTask::default(),
                        episode: 0,
                        recorder: None,
                        recording_period: None,
                        telemetry: None,
                    })
                }));
                let mut session = match initialized {
                    Ok(Ok(session)) => {
                        let _ = ready_tx.send(Ok(()));
                        session
                    }
                    Ok(Err(error)) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                    Err(_) => {
                        let _ = ready_tx.send(Err(stopped()));
                        return;
                    }
                };
                for job in receiver {
                    if catch_unwind(AssertUnwindSafe(|| job(&mut session))).is_err() {
                        break;
                    }
                }
            })?;
        ready_rx.recv().map_err(|_| stopped())??;
        Ok(Self {
            sender: Mutex::new(Some(sender)),
            handle: Mutex::new(Some(handle)),
        })
    }
    pub(crate) fn access<T: Send + 'static>(
        &self,
        f: impl FnOnce(&mut Session) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| stopped())?
            .clone()
            .ok_or_else(stopped)?;
        let (tx, rx) = mpsc::channel();
        sender
            .send(Box::new(move |session| {
                let _ = tx.send(f(session));
            }))
            .map_err(|_| stopped())?;
        rx.recv().map_err(|_| stopped())?
    }
    pub(crate) fn close(&self) -> Result<()> {
        self.sender.lock().map_err(|_| stopped())?.take();
        if let Some(handle) = self.handle.lock().map_err(|_| stopped())?.take() {
            handle.join().map_err(|_| stopped())?;
        }
        Ok(())
    }
}
// Dropping Python objects must not block with the GIL held. Closing the channel
// lets the owning worker destroy MuJoCo allocations; close() explicitly joins
// while detached. Dropping a JoinHandle merely detaches it.
impl Drop for Worker {
    fn drop(&mut self) {
        if let Ok(sender) = self.sender.get_mut() {
            sender.take();
        }
    }
}
