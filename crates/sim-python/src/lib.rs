mod pick_place;
mod worker;
use mujoco_runtime::{Error, JointCommand, JointState, Simulator as CoreSimulator};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use robotics_core::task::Task;
use std::{
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
};
use worker::Worker;
type State = HashMap<&'static str, Vec<f64>>;
#[pyclass(name = "Simulator", module = "robot_sim._native", frozen)]
struct Simulator {
    inner: Worker,
}
#[pymethods]
impl Simulator {
    #[new]
    #[pyo3(signature = (model_dir=None, *, robot=None))]
    fn new(py: Python<'_>, model_dir: Option<PathBuf>, robot: Option<&str>) -> PyResult<Self> {
        if model_dir.is_some() && robot.is_some() {
            return Err(PyValueError::new_err(
                "model_dir and robot are mutually exclusive",
            ));
        }
        let robot = robot.unwrap_or("so101").to_owned();
        boundary(|| py.detach(move || Worker::new(model_dir, robot))).map(|inner| Self { inner })
    }
    fn configure_reach_selection(&self, py: Python<'_>, names: Vec<String>) -> PyResult<()> {
        self.session(py, move |s| {
            let all = s.sim.actuator_names();
            if names.is_empty()
                || names.iter().any(|n| !all.contains(n))
                || names.iter().collect::<std::collections::HashSet<_>>().len() != names.len()
            {
                return Err(Error::Input("invalid controlled joint selection".into()));
            }
            let home = s.sim.home_position();
            s.task.set_fixed_joints(
                all.iter()
                    .enumerate()
                    .filter(|(_, n)| !names.contains(n))
                    .map(|(i, _)| (i, home[i]))
                    .collect(),
            );
            Ok(())
        })
    }
    #[pyo3(signature=(position,target,indices,damping=0.01,max_delta=0.1))]
    fn dls_delta(
        &self,
        py: Python<'_>,
        position: Vec<f64>,
        target: [f64; 3],
        indices: Vec<usize>,
        damping: f64,
        max_delta: f64,
    ) -> PyResult<Vec<f64>> {
        self.access(py, move |s| {
            s.dls_delta(&position, target, &indices, damping, max_delta)
        })
    }
    fn forward_kinematics(&self, py: Python<'_>, position: Vec<f64>) -> PyResult<Vec<f64>> {
        self.access(py, move |s| {
            s.forward_kinematics(&position, s.default_end_effector()?)
                .map(|p| p.to_vec())
        })
    }
    fn configure_randomization(&self, py: Python<'_>, config_json: String) -> PyResult<()> {
        self.access(py, move |s| {
            let config =
                serde_json::from_str(&config_json).map_err(|e| Error::Input(e.to_string()))?;
            s.configure_randomization(config)
        })
    }
    fn set_parameter_center(&self, py: Python<'_>, parameters_json: String) -> PyResult<()> {
        self.access(py, move |s| {
            s.set_parameter_center(
                serde_json::from_str(&parameters_json).map_err(|e| Error::Input(e.to_string()))?,
            )
        })
    }
    fn load_parameters(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        self.access(py, move |s| {
            s.set_parameter_center(mujoco_runtime::randomization::ModelParameters::load(path)?)
        })
    }
    fn save_parameters(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        self.access(py, move |s| s.parameters().save(path))
    }
    fn parameters_json(&self, py: Python<'_>) -> PyResult<String> {
        self.access(py, |s| {
            serde_json::to_string(s.parameters()).map_err(|e| Error::Input(e.to_string()))
        })
    }
    fn start_trajectory(
        &self,
        py: Python<'_>,
        directory: PathBuf,
        control_period: f64,
        seed: u64,
    ) -> PyResult<()> {
        self.session(py, move |s| {
            use robotics_core::trajectory::TrajectoryRecorder;
            if s.recorder.is_some() {
                return Err(Error::Input("recording already active".into()));
            }
            if !control_period.is_finite() || control_period < s.sim.timestep() {
                return Err(Error::Input(
                    "recording period must be finite and >= physics timestep".into(),
                ));
            }
            let metadata = s.sim.trajectory_metadata(control_period, seed)?;
            let mut recorder = trajectory_io::CsvRecorder::new(directory);
            recorder.begin(&metadata)?;
            s.recorder = Some(Box::new(recorder));
            s.recording_period = Some(control_period);
            Ok(())
        })
    }
    fn finish_trajectory(&self, py: Python<'_>) -> PyResult<()> {
        self.session(py, |s| s.finish_recording())
    }
    #[getter]
    fn home_position(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        self.access(py, |s| Ok(s.home_position()))
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        boundary(|| py.detach(|| self.inner.close()))
    }
    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        self.session(py, |s| {
            s.finish_recording()?;
            s.sim.reset()
        })
    }
    fn state(&self, py: Python<'_>) -> PyResult<State> {
        self.access(py, |sim| Ok(python_state(sim.state())))
    }
    fn step(&self, py: Python<'_>, position: Vec<f64>) -> PyResult<State> {
        self.session(py, move |s| {
            let command = JointCommand { position };
            s.check_recording_period(s.sim.timestep())?;
            let state = s.sim.step(&command)?;
            s.task.advance();
            s.log(&command)?;
            Ok(python_state(state))
        })
    }
    fn step_control(&self, py: Python<'_>, position: Vec<f64>, period: f64) -> PyResult<State> {
        self.session(py, move |s| {
            let command = JointCommand { position };
            s.check_recording_period(period)?;
            let state = s.sim.step_control(&command, period)?;
            s.task.advance();
            s.log(&command)?;
            Ok(python_state(state))
        })
    }
    fn task_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let result = self.session(py, |s| s.snapshot())?;
        task_dict(py, result)
    }
    fn step_task<'py>(
        &self,
        py: Python<'py>,
        position: Vec<f64>,
        period: f64,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let result = self.session(py, move |s| {
            let command = JointCommand { position };
            s.check_recording_period(period)?;
            s.sim.step_control(&command, period)?;
            s.task.advance();
            s.log(&command)?;
            s.snapshot()
        })?;
        task_dict(py, result)
    }
    #[pyo3(signature = (record=None))]
    fn start_rerun(&self, py: Python<'_>, record: Option<PathBuf>) -> PyResult<()> {
        #[cfg(feature = "rerun")]
        return self.session(py, move |s| {
            let sink = match record {
                Some(path) => telemetry_rerun::RerunTelemetry::save(path),
                None => telemetry_rerun::RerunTelemetry::spawn(),
            }?;
            s.telemetry = Some(Box::new(sink));
            s.telemetry.as_mut().unwrap().begin_episode(s.episode)?;
            s.log(&JointCommand {
                position: s.sim.state().position,
            })
        });
        #[cfg(not(feature = "rerun"))]
        {
            let _ = (py, record);
            Err(PyRuntimeError::new_err(
                "install with maturin develop --features sim-python/rerun",
            ))
        }
    }
    fn flush_telemetry(&self, py: Python<'_>) -> PyResult<()> {
        self.session(py, |s| {
            if let Some(sink) = &mut s.telemetry {
                sink.flush()?;
            }
            Ok(())
        })
    }
    fn reset_reach(&self, py: Python<'_>, seed: u64) -> PyResult<()> {
        self.session(py, move |s| {
            s.finish_recording()?;
            s.sim.reset_seed(seed)?;
            s.task.reset(&s.sim, seed)?;
            s.episode += 1;
            if let Some(sink) = &mut s.telemetry {
                sink.begin_episode(s.episode)?;
            }
            s.log(&JointCommand {
                position: s.sim.state().position,
            })
        })
    }
    #[pyo3(signature=(*, success_threshold=0.02, success_bonus=0.0, max_episode_steps=1000, min_target_distance=0.05, max_target_distance=0.25, joint_margin=0.05, max_target_attempts=4096))]
    #[allow(clippy::too_many_arguments)]
    fn configure_reach(
        &self,
        py: Python<'_>,
        success_threshold: f64,
        success_bonus: f64,
        max_episode_steps: usize,
        min_target_distance: f64,
        max_target_distance: f64,
        joint_margin: f64,
        max_target_attempts: usize,
    ) -> PyResult<()> {
        self.session(py, move |s| {
            let fixed = s.task.fixed_joints().to_vec();
            s.task = robotics_core::task::ReachTask::new(robotics_core::task::ReachConfig {
                success_threshold,
                success_bonus,
                max_episode_steps,
                min_target_distance,
                max_target_distance,
                joint_margin,
                max_target_attempts,
            })?;
            s.task.set_fixed_joints(fixed);
            Ok(())
        })
    }
    #[getter]
    fn distance(&self, py: Python<'_>) -> PyResult<f64> {
        self.session(py, |s| s.task.distance(&s.sim))
    }
    #[getter]
    fn truncated(&self, py: Python<'_>) -> PyResult<bool> {
        self.session(py, |s| s.task.truncated(&s.sim))
    }
    #[getter]
    fn episode_step(&self, py: Python<'_>) -> PyResult<usize> {
        self.session(py, |s| Ok(s.task.episode_steps()))
    }
    #[getter]
    fn reward(&self, py: Python<'_>) -> PyResult<f64> {
        self.session(py, |s| s.task.reward(&s.sim))
    }
    #[getter]
    fn terminated(&self, py: Python<'_>) -> PyResult<bool> {
        self.session(py, |s| s.task.terminated(&s.sim))
    }
    #[getter]
    fn target(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        self.session(py, |s| Ok(s.task.target().to_vec()))
    }
    #[getter]
    fn end_effector(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        self.access(py, |s| {
            s.end_effector(s.default_end_effector()?)
                .map(|p| p.to_vec())
        })
    }
    #[getter]
    fn action_bounds(&self, py: Python<'_>) -> PyResult<Vec<[f64; 2]>> {
        self.access(py, |s| Ok(s.action_bounds()))
    }
    #[getter]
    fn joint_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.access(py, |sim| {
            Ok(sim
                .spec()
                .actuators
                .iter()
                .map(|a| a.joint.clone())
                .collect())
        })
    }
    #[getter]
    fn camera_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.access(py, |sim| {
            Ok(sim.spec().cameras.iter().map(|c| c.name.clone()).collect())
        })
    }
    #[getter]
    fn time(&self, py: Python<'_>) -> PyResult<f64> {
        self.access(py, |sim| Ok(sim.time()))
    }
    #[getter]
    fn timestep(&self, py: Python<'_>) -> PyResult<f64> {
        self.access(py, |sim| Ok(sim.timestep()))
    }
    #[getter]
    fn actuator_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.access(py, |sim| Ok(sim.actuator_names().to_vec()))
    }
}
impl Simulator {
    fn session<T: Send + 'static>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut worker::Session) -> robotics_core::Result<T> + Send + 'static,
    ) -> PyResult<T> {
        boundary(|| py.detach(|| self.inner.access(f)))
    }
    fn access<T: Send + 'static>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut CoreSimulator) -> robotics_core::Result<T> + Send + 'static,
    ) -> PyResult<T> {
        self.session(py, move |s| f(&mut s.sim))
    }
}
fn boundary<T>(f: impl FnOnce() -> robotics_core::Result<T>) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err(Error::Worker("Rust simulator panicked".into())))
        .map_err(python_error)
}
fn python_state(state: JointState) -> State {
    HashMap::from([("position", state.position), ("velocity", state.velocity)])
}
fn python_error(error: Error) -> PyErr {
    match error {
        Error::Command { .. } | Error::Dimension { .. } | Error::Input(_) => {
            PyValueError::new_err(error.to_string())
        }
        Error::Io(source) => source.into(),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
#[pyclass(name = "SineTrajectory")]
struct PySineTrajectory {
    inner: robotics_core::controller::SineTrajectory,
}
#[pymethods]
impl PySineTrajectory {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        home: Vec<f64>,
        bounds: Vec<[f64; 2]>,
        joint_names: Vec<String>,
        joint: String,
        amplitude: f64,
        frequency: f64,
        duration: f64,
    ) -> PyResult<Self> {
        let index = joint_names
            .iter()
            .position(|n| n == &joint)
            .ok_or_else(|| PyValueError::new_err("unknown excitation joint"))?;
        boundary(|| {
            robotics_core::controller::SineTrajectory::new(
                home, &bounds, index, amplitude, frequency, duration,
            )
        })
        .map(|inner| Self { inner })
    }
    fn sample(&mut self, time: f64) -> PyResult<Vec<f64>> {
        boundary(|| self.inner.sample(time).map(|c| c.position))
    }
}
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Simulator>()?;
    module.add_class::<pick_place::PickPlaceSimulator>()?;
    module.add_class::<pick_place::PickPlaceRenderer>()?;
    module.add("GRIPPER_OPEN", mujoco_runtime::skills::GRIPPER_OPEN)?;
    module.add("GRIPPER_CLOSED", mujoco_runtime::skills::GRIPPER_CLOSED)?;
    module.add("CONTROL_PERIOD", mujoco_runtime::skills::CONTROL_PERIOD)?;
    module.add_class::<PySineTrajectory>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add(
        "ACTUATOR_NAMES",
        mujoco_runtime::presets::SO101_ACTUATORS.to_vec(),
    )?;
    Ok(())
}
#[cfg(all(test, not(feature = "extension-module")))]
mod tests {
    use super::*;
    #[test]
    fn converts_rust_panic_to_runtime_error() {
        Python::initialize();
        Python::attach(|py| {
            let error = boundary::<()>(|| panic!("injected failure")).unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
        });
    }
    #[test]
    fn channel_wait_detaches_from_python() {
        Python::initialize();
        Python::attach(|py| {
            let sim = Simulator::new(py, None, Some("so101")).unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            let other = std::thread::spawn(move || {
                Python::attach(|_| {
                    tx.send(()).unwrap();
                })
            });
            // The other thread cannot attach until the waiting caller releases
            // the GIL. Timeout makes a regression fail rather than deadlock.
            sim.session(py, move |_| {
                rx.recv_timeout(std::time::Duration::from_secs(5))
                    .map_err(|e| Error::Worker(e.to_string()))?;
                Ok(())
            })
            .unwrap();
            py.detach(|| other.join()).unwrap();
            sim.close(py).unwrap();
        });
    }
    #[test]
    fn refuses_to_reuse_instance_after_panic() {
        Python::initialize();
        Python::attach(|py| {
            let sim = Simulator::new(py, None, Some("so101")).unwrap();
            let error = sim
                .access::<()>(py, |_| panic!("injected worker failure"))
                .unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert!(sim.state(py).unwrap_err().to_string().contains("poisoned"));
        });
    }
}

fn task_dict(
    py: Python<'_>,
    r: robotics_core::task::StepResult,
) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("joint_position", r.joint_state.position)?;
    d.set_item("joint_velocity", r.joint_state.velocity)?;
    d.set_item("end_effector_position", r.end_effector_position)?;
    d.set_item("target_position", r.target_position)?;
    d.set_item("reward", r.reward)?;
    d.set_item("distance", r.distance)?;
    d.set_item("terminated", r.terminated)?;
    d.set_item("truncated", r.truncated)?;
    d.set_item("time", r.simulation_time)?;
    d.set_item("observation_time", r.observation_time)?;
    d.set_item("episode_step", r.episode_step)?;
    Ok(d)
}
