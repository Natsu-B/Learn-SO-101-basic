//! Python pick-and-place API. Physics and skill decisions stay in the runtime;
//! only owned snapshots cross the worker channel.
use crate::{boundary, worker::Worker};
use mujoco_runtime::{
    Error, JointCommand,
    pick_place::{GRIPPER, ObjectRandomization, PickPlaceConfig, PickPlaceSim, PickPlaceState},
    presets,
    skills::{
        self, CONTROL_PERIOD, Criteria, EpisodeReport, ScriptedGrasp, ScriptedPlace,
        ScriptedTransport, Skill, SkillReport, SkillStatus, predicates,
    },
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyDict},
};
use robotics_core::Result;
use std::{collections::HashMap, path::PathBuf};

struct ActiveSkill {
    skill: Box<dyn Skill>,
    steps: usize,
    status: SkillStatus,
}
impl ActiveSkill {
    fn begin(env: &PickPlaceSim, name: &str, max_steps: Option<usize>) -> Result<Self> {
        let criteria = Criteria::default();
        let mut skill: Box<dyn Skill> = match name {
            "grasp" => {
                let mut skill = ScriptedGrasp::new(criteria);
                if let Some(n) = max_steps {
                    skill.max_steps = n;
                }
                Box::new(skill)
            }
            "transport" => {
                let mut skill = ScriptedTransport::new(criteria);
                if let Some(n) = max_steps {
                    skill.max_steps = n;
                }
                Box::new(skill)
            }
            "place" => {
                let mut skill = ScriptedPlace::new(criteria);
                if let Some(n) = max_steps {
                    skill.max_steps = n;
                }
                Box::new(skill)
            }
            _ => {
                return Err(Error::Input(
                    "skill must be grasp, transport or place".into(),
                ));
            }
        };
        skill.begin(env)?;
        let status = skill.status(env);
        Ok(Self {
            skill,
            steps: 0,
            status,
        })
    }
    fn step(&mut self, env: &mut PickPlaceSim) -> Result<()> {
        if self.status == SkillStatus::Running {
            let command = self.skill.act(env)?;
            env.step(&command, CONTROL_PERIOD)?;
            self.steps += 1;
            self.status = self.skill.status(env);
        }
        Ok(())
    }
    fn report(&self) -> SkillReport {
        SkillReport {
            name: self.skill.name().into(),
            learned: self.skill.learned(),
            success: self.status == SkillStatus::Success,
            steps: self.steps,
            reason: match &self.status {
                SkillStatus::Failure(reason) => Some(reason.clone()),
                _ => None,
            },
        }
    }
}

struct Session {
    sim: PickPlaceSim,
    active: Option<ActiveSkill>,
}
struct Snapshot {
    state: PickPlaceState,
    cube_vel: [f64; 6],
}
impl Session {
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            state: self.sim.state(),
            cube_vel: self.sim.cube_velocity(),
        }
    }
}

#[pyclass(module = "robot_sim._native", frozen)]
pub(crate) struct PickPlaceSimulator {
    inner: Worker<Session>,
}
#[pymethods]
impl PickPlaceSimulator {
    #[new]
    #[pyo3(signature = (model_dir=None, *, robot=None))]
    fn new(py: Python<'_>, model_dir: Option<PathBuf>, robot: Option<&str>) -> PyResult<Self> {
        if model_dir.is_some() && robot.is_some() {
            return Err(PyValueError::new_err(
                "model_dir and robot are mutually exclusive",
            ));
        }
        let robot = robot.unwrap_or("so101").to_owned();
        boundary(|| {
            py.detach(move || {
                Worker::spawn(move || {
                    if robot != "so101" {
                        return Err(Error::Input(
                            "PickPlaceSimulator supports only so101".into(),
                        ));
                    }
                    let spec = match model_dir {
                        Some(dir) => presets::so101(dir),
                        None => presets::robot(&robot)?,
                    };
                    Ok(Session {
                        sim: PickPlaceSim::from_spec(spec, PickPlaceConfig::default())?,
                        active: None,
                    })
                })
            })
        })
        .map(|inner| Self { inner })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        boundary(|| py.detach(|| self.inner.close()))
    }
    fn reset(&self, py: Python<'_>, seed: u64) -> PyResult<()> {
        self.session(py, move |s| {
            s.active = None;
            s.sim.reset(seed)
        })
    }
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        state_dict(py, self.session(py, |s| Ok(s.snapshot()))?)
    }
    /// Advance physics using the runtime's command validation and control clock.
    /// A manual step cancels any skill started by start_skill().
    #[pyo3(signature = (position, period=CONTROL_PERIOD))]
    fn step<'py>(
        &self,
        py: Python<'py>,
        position: Vec<f64>,
        period: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = self.session(py, move |s| {
            s.sim.step(&JointCommand { position }, period)?;
            s.active = None;
            Ok(s.snapshot())
        })?;
        state_dict(py, snapshot)
    }
    /// Object mass (kg) / sliding friction ranges for subsequent resets.
    /// None keeps the scene value; equal ends fix it. Independent of joint DR.
    #[pyo3(signature = (*, mass=None, friction=None))]
    fn set_object_randomization(
        &self,
        py: Python<'_>,
        mass: Option<[f64; 2]>,
        friction: Option<[f64; 2]>,
    ) -> PyResult<()> {
        self.session(py, move |s| {
            s.sim
                .set_object_randomization(ObjectRandomization { mass, friction })
        })
    }
    /// Ground-truth object physics of this episode. For logs only: policy
    /// observations must not include it.
    fn object_physics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (p, nominal, r) = self.session(py, |s| {
            Ok((
                s.sim.object_physics(),
                s.sim.nominal_object_physics(),
                s.sim.config().object_randomization,
            ))
        })?;
        let d = PyDict::new(py);
        d.set_item("mass", p.mass)?;
        d.set_item("inertia", p.inertia.to_vec())?;
        d.set_item("friction", p.friction)?;
        d.set_item("priority", p.priority)?;
        d.set_item("nominal_mass", nominal.mass)?;
        d.set_item("nominal_friction", nominal.friction)?;
        d.set_item("mass_range", r.mass.map(|v| v.to_vec()))?;
        d.set_item("friction_range", r.friction.map(|v| v.to_vec()))?;
        Ok(d)
    }
    fn action_bounds(&self, py: Python<'_>) -> PyResult<Vec<[f64; 2]>> {
        self.session(py, |s| Ok(s.sim.sim.action_bounds()))
    }
    fn joint_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.session(py, |s| {
            Ok(s.sim
                .sim
                .spec()
                .actuators
                .iter()
                .map(|a| a.joint.clone())
                .collect())
        })
    }
    fn timestep(&self, py: Python<'_>) -> PyResult<f64> {
        self.session(py, |s| Ok(s.sim.sim.timestep()))
    }
    /// Instantaneous runtime predicates; grasp's hold duration is enforced by the skill.
    fn predicates(&self, py: Python<'_>) -> PyResult<HashMap<&'static str, bool>> {
        self.session(py, |s| {
            let c = Criteria::default();
            Ok(HashMap::from([
                ("lifted", predicates::lifted(&s.sim, &c)),
                ("transported", predicates::transported(&s.sim, &c)),
                ("placed", predicates::placed(&s.sim, &c)),
                ("dropped", predicates::dropped(&s.sim)),
            ]))
        })
    }
    /// Default success criteria and scene/skill heights, in metres and seconds.
    fn criteria(&self, py: Python<'_>) -> PyResult<HashMap<&'static str, f64>> {
        self.session(py, |s| {
            let c = Criteria::default();
            let carry_height = ScriptedTransport::new(c.clone()).carry_height;
            Ok(HashMap::from([
                ("lift_height", c.lift_height),
                ("hold_seconds", c.hold_seconds),
                ("transport_tolerance", c.transport_tolerance),
                ("place_tolerance", c.place_tolerance),
                ("rest_speed", c.rest_speed),
                ("cube_half", s.sim.config().cube_half),
                ("carry_height", carry_height),
            ]))
        })
    }
    /// Begin a fresh scripted skill from the current physics state.
    fn start_skill(&self, py: Python<'_>, name: String) -> PyResult<()> {
        self.session(py, move |s| {
            s.active = Some(ActiveSkill::begin(&s.sim, &name, None)?);
            Ok(())
        })
    }
    /// Return running/success/failure. Terminal calls leave physics unchanged.
    fn skill_step(&self, py: Python<'_>) -> PyResult<&'static str> {
        self.session(py, |s| {
            let skill = s
                .active
                .as_mut()
                .ok_or_else(|| Error::Input("start_skill first".into()))?;
            skill.step(&mut s.sim)?;
            Ok(status_name(&skill.status))
        })
    }
    /// Inspect the active scripted skill without restarting it or advancing physics.
    fn skill_report<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (report, status) = self.session(py, |s| {
            let skill = s
                .active
                .as_ref()
                .ok_or_else(|| Error::Input("start_skill first".into()))?;
            Ok((skill.report(), status_name(&skill.status)))
        })?;
        let dict = skill_dict(py, report)?;
        dict.set_item("status", status)?;
        Ok(dict)
    }
    #[pyo3(signature = (name, max_steps=None))]
    fn run_skill<'py>(
        &self,
        py: Python<'py>,
        name: String,
        max_steps: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let report = self.session(py, move |s| {
            let mut skill = ActiveSkill::begin(&s.sim, &name, max_steps)?;
            while skill.status == SkillStatus::Running {
                skill.step(&mut s.sim)?;
            }
            let report = skill.report();
            s.active = Some(skill);
            Ok(report)
        })?;
        skill_dict(py, report)
    }
    fn run_episode<'py>(&self, py: Python<'py>, seed: u64) -> PyResult<Bound<'py, PyDict>> {
        let report = self.session(py, move |s| {
            s.active = None;
            skills::run_episode(
                &mut s.sim,
                &mut skills::scripted_skills(&Criteria::default()),
                seed,
                |_, _| {},
            )
        })?;
        episode_dict(py, report)
    }
}

/// Main-thread visualization of a private compiled-model/data copy.
#[pyclass(module = "robot_sim._native", unsendable)]
pub(crate) struct PickPlaceRenderer {
    renderer: Option<mujoco_runtime::render::OffscreenRenderer>,
    source: Py<PickPlaceSimulator>,
}
#[pymethods]
impl PickPlaceRenderer {
    #[new]
    #[pyo3(signature = (sim, width=1280, height=720))]
    fn new(py: Python<'_>, sim: Py<PickPlaceSimulator>, width: u32, height: u32) -> PyResult<Self> {
        let model = sim.borrow(py).session(py, |s| {
            Ok(mujoco_runtime::render::model_snapshot(&s.sim.sim))
        })?;
        let renderer =
            boundary(|| mujoco_runtime::render::OffscreenRenderer::new(&model, width, height))?;
        Ok(Self {
            renderer: Some(renderer),
            source: sim,
        })
    }

    /// Return width * height * 3 RGB bytes; source physics is read-only.
    fn render<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let renderer = self
            .renderer
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("renderer is closed"))?;
        let state = self.source.borrow(py).session(py, |s| {
            Ok(mujoco_runtime::render::state_snapshot(&s.sim.sim))
        })?;
        let bytes = boundary(|| renderer.render(&state))?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn close(&mut self) {
        self.renderer = None;
    }
}
impl PickPlaceSimulator {
    fn session<T: Send + 'static>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut Session) -> Result<T> + Send + 'static,
    ) -> PyResult<T> {
        boundary(|| py.detach(|| self.inner.access(f)))
    }
}

fn status_name(status: &SkillStatus) -> &'static str {
    match status {
        SkillStatus::Running => "running",
        SkillStatus::Success => "success",
        SkillStatus::Failure(_) => "failure",
    }
}
fn state_dict(py: Python<'_>, snapshot: Snapshot) -> PyResult<Bound<'_, PyDict>> {
    let s = snapshot.state;
    let d = PyDict::new(py);
    d.set_item("time", s.time)?;
    d.set_item("gripper", s.joint_position[GRIPPER])?;
    d.set_item("qpos", s.joint_position)?;
    d.set_item("qvel", s.joint_velocity)?;
    d.set_item("cube_pos", s.cube_position.to_vec())?;
    d.set_item("cube_quat", s.cube_quat.to_vec())?;
    d.set_item("cube_vel", snapshot.cube_vel.to_vec())?;
    d.set_item("target", s.target_position.to_vec())?;
    d.set_item("grasp_center", s.gripper_position.to_vec())?;
    d.set_item(
        "contacts",
        HashMap::from([
            ("fixed_jaw", s.contacts.fixed_jaw),
            ("moving_jaw", s.contacts.moving_jaw),
            ("table", s.contacts.table),
            ("other_robot", s.contacts.other_robot),
            ("pinched", s.contacts.pinched()),
        ]),
    )?;
    Ok(d)
}
fn skill_dict(py: Python<'_>, r: SkillReport) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("name", r.name)?;
    d.set_item("status", if r.success { "success" } else { "failure" })?;
    d.set_item("success", r.success)?;
    d.set_item("learned", r.learned)?;
    d.set_item("steps", r.steps)?;
    d.set_item("reason", r.reason)?;
    Ok(d)
}
fn episode_dict(py: Python<'_>, r: EpisodeReport) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("seed", r.seed)?;
    d.set_item("cube_initial", r.cube_initial.to_vec())?;
    d.set_item("target", r.target.to_vec())?;
    d.set_item("cube_final", r.cube_final.to_vec())?;
    d.set_item("final_xy_error", r.final_xy_error)?;
    d.set_item("success", r.success)?;
    d.set_item("control_steps", r.control_steps)?;
    let reports = r
        .skills
        .into_iter()
        .map(|r| skill_dict(py, r))
        .collect::<PyResult<Vec<_>>>()?;
    d.set_item("skills", reports)?;
    Ok(d)
}

#[cfg(all(test, not(feature = "extension-module")))]
mod tests {
    use super::*;
    use pyo3::exceptions::PyRuntimeError;

    #[test]
    fn panic_poisons_pick_place_worker() {
        Python::initialize();
        Python::attach(|py| {
            let sim = PickPlaceSimulator::new(py, None, None).unwrap();
            let error = sim
                .session::<()>(py, |_| panic!("injected pick-place failure"))
                .unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert!(sim.state(py).unwrap_err().to_string().contains("poisoned"));
            sim.close(py).unwrap();
        });
    }

    #[test]
    fn pick_place_wait_releases_gil() {
        Python::initialize();
        Python::attach(|py| {
            let sim = PickPlaceSimulator::new(py, None, None).unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            let other = std::thread::spawn(move || Python::attach(|_| tx.send(()).unwrap()));
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
}
