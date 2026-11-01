//! Python API of the compliant-object grasp-force environment
//! (docs/FORCE_CONTROL.md). Physics, contact scans and the force-PI baseline
//! stay in the runtime; only owned snapshots cross the worker channel.
use crate::{boundary, worker::Worker};
use mujoco_runtime::force_grasp::{
    CompliantObject, ForceGraspRanges, ForceGraspSim, ForcePi, ForceReading, PENETRATION_LIMIT,
    Positioning, StepStats,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use robotics_core::Result;

fn reading_dict<'py>(py: Python<'py>, r: &ForceReading) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("time", r.time)?;
    d.set_item("gripper_q", r.gripper_q)?;
    d.set_item("gripper_qvel", r.gripper_qvel)?;
    d.set_item("gripper_command", r.gripper_command)?;
    d.set_item("actuator_torque", r.actuator_torque)?;
    d.set_item("torque_limit", r.torque_limit)?;
    d.set_item("jaw_force", r.jaw_force.to_vec())?;
    d.set_item("rigid_force", r.rigid_force.to_vec())?;
    d.set_item("deformation", r.deformation.to_vec())?;
    d.set_item("spring_force", r.spring_force.to_vec())?;
    d.set_item("bottomed", r.bottomed.to_vec())?;
    d.set_item("penetration", r.penetration)?;
    d.set_item("table_force", r.table_force)?;
    d.set_item("object_position", r.object_position.to_vec())?;
    d.set_item("object_velocity", r.object_velocity.to_vec())?;
    d.set_item("object_closing_velocity", r.object_closing_velocity)?;
    d.set_item("object_displacement", r.object_displacement)?;
    d.set_item("fixed_face_shift", r.fixed_face_shift)?;
    Ok(d)
}
fn stats_dict<'py>(py: Python<'py>, s: &StepStats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("physics_steps", s.physics_steps)?;
    d.set_item("max_jaw_force", s.max_jaw_force.to_vec())?;
    d.set_item("min_jaw_force", s.min_jaw_force.to_vec())?;
    d.set_item("max_rigid_force", s.max_rigid_force.to_vec())?;
    d.set_item("max_deformation", s.max_deformation.to_vec())?;
    d.set_item("bottomed", s.bottomed.to_vec())?;
    d.set_item("max_penetration", s.max_penetration)?;
    d.set_item("max_actuator_torque", s.max_actuator_torque)?;
    Ok(d)
}
fn object_dict<'py>(py: Python<'py>, o: &CompliantObject) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("width", o.width)?;
    d.set_item("gap", o.gap)?;
    d.set_item("stiffness", o.stiffness)?;
    d.set_item("damping", o.damping)?;
    d.set_item("travel", o.travel)?;
    d.set_item("friction", o.friction)?;
    d.set_item("mass", o.mass)?;
    Ok(d)
}
fn object_from(d: &Bound<'_, PyDict>) -> PyResult<CompliantObject> {
    let mut o = CompliantObject::nominal();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let v: f64 = v.extract()?;
        match key.as_str() {
            "width" => o.width = v,
            "gap" => o.gap = v,
            "stiffness" => o.stiffness = v,
            "damping" => o.damping = v,
            "travel" => o.travel = v,
            "friction" => o.friction = v,
            "mass" => o.mass = v,
            _ => return Err(PyValueError::new_err(format!("unknown object field {key}"))),
        }
    }
    Ok(o)
}

/// Grasp-force environment with a compliant object. ``reading()`` and the
/// per-step ``stats`` include ground truth (deformation, spring force) for
/// evaluation; policy observations must use only gripper state, jaw forces
/// and the goal.
#[pyclass(module = "robot_sim._native", frozen)]
pub(crate) struct ForceGraspSimulator {
    inner: Worker<ForceGraspSim>,
}
#[pymethods]
impl ForceGraspSimulator {
    /// ``positioning="teleport"`` solves the scripted descent target by IK
    /// once and places the arm there (fast resets for RL); ``"scripted"``
    /// runs the scripted Grasp approach (docs §3–4).
    #[new]
    #[pyo3(signature = (*, positioning="scripted"))]
    fn new(py: Python<'_>, positioning: &str) -> PyResult<Self> {
        let positioning = match positioning {
            "scripted" => Positioning::Scripted,
            "teleport" => Positioning::Teleport,
            _ => {
                return Err(PyValueError::new_err(
                    "positioning must be scripted or teleport",
                ));
            }
        };
        boundary(|| {
            py.detach(move || {
                Worker::spawn(move || {
                    let mut sim = ForceGraspSim::new(ForceGraspRanges::default())?;
                    sim.positioning = positioning;
                    Ok(sim)
                })
            })
        })
        .map(|inner| Self { inner })
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        boundary(|| py.detach(|| self.inner.close()))
    }
    /// Sampling ranges as a dict of [lo, hi] pairs.
    fn ranges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = self.session(py, |s| Ok(*s.ranges()))?;
        let d = PyDict::new(py);
        for (k, v) in ranges_items(&r) {
            d.set_item(k, v.to_vec())?;
        }
        Ok(d)
    }
    /// Change sampling ranges for later resets; omitted keys are kept.
    #[pyo3(signature = (**kwargs))]
    fn set_ranges(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let mut updates = Vec::new();
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                let key: String = k.extract()?;
                let v: [f64; 2] = v.extract()?;
                if !v.iter().all(|x| x.is_finite()) {
                    return Err(PyValueError::new_err(format!("non-finite range {key}")));
                }
                updates.push((key, v));
            }
        }
        self.session(py, move |s| {
            let mut r = *s.ranges();
            for (key, v) in updates {
                let slot = match key.as_str() {
                    "width" => &mut r.width,
                    "gap" => &mut r.gap,
                    "kappa" => &mut r.kappa,
                    "stiffness" => &mut r.stiffness,
                    "damping_ratio" => &mut r.damping_ratio,
                    "travel" => &mut r.travel,
                    "friction" => &mut r.friction,
                    "mass" => &mut r.mass,
                    "gripper_damping_scale" => &mut r.gripper_damping_scale,
                    _ => {
                        return Err(robotics_core::Error::Input(format!("unknown range {key}")));
                    }
                };
                *slot = v;
            }
            s.set_ranges(r);
            Ok(())
        })
    }
    /// Sample (or set) the object, position the open gripper and freeze the arm.
    #[pyo3(signature = (seed, object=None))]
    fn reset<'py>(
        &self,
        py: Python<'py>,
        seed: u64,
        object: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let object = object.map(object_from).transpose()?;
        let r = self.session(py, move |s| {
            s.reset(seed, object)?;
            s.reading()
        })?;
        reading_dict(py, &r)
    }
    /// One 30 Hz control period with an absolute gripper target (rad).
    /// Returns {"reading": ..., "stats": per-physics-step extremes}.
    fn step<'py>(&self, py: Python<'py>, gripper: f64) -> PyResult<Bound<'py, PyDict>> {
        let (r, st) = self.session(py, move |s| {
            let st = s.step(gripper)?;
            Ok((s.reading()?, st))
        })?;
        let d = PyDict::new(py);
        d.set_item("reading", reading_dict(py, &r)?)?;
        d.set_item("stats", stats_dict(py, &st)?)?;
        Ok(d)
    }
    fn reading<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = self.session(py, |s| s.reading())?;
        reading_dict(py, &r)
    }
    /// Ground-truth object of this episode. For logs only.
    fn object<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let o = self.session(py, |s| Ok(s.object()))?;
        object_dict(py, &o)
    }
    /// [f_min, f_max] the current object can bear (f_max <= f_cap).
    fn goal_range(&self, py: Python<'_>, f_min: f64, f_cap: f64) -> PyResult<[f64; 2]> {
        self.session(py, move |s| Ok(s.object().goal_range(f_min, f_cap)))
    }
    fn gripper_bounds(&self, py: Python<'_>) -> PyResult<[f64; 2]> {
        self.session(py, |s| Ok(s.gripper_bounds()))
    }
    /// (object displacement while positioning [m], gripper damping scale).
    fn episode_info(&self, py: Python<'_>) -> PyResult<(f64, f64)> {
        self.session(py, |s| {
            Ok((s.positioning_displacement, s.gripper_damping_scale))
        })
    }
    #[classattr]
    fn penetration_limit() -> f64 {
        PENETRATION_LIMIT
    }
}
impl ForceGraspSimulator {
    fn session<T: Send + 'static>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut ForceGraspSim) -> Result<T> + Send + 'static,
    ) -> PyResult<T> {
        boundary(|| py.detach(|| self.inner.access(f)))
    }
}
fn ranges_items(r: &ForceGraspRanges) -> [(&'static str, [f64; 2]); 9] {
    [
        ("width", r.width),
        ("gap", r.gap),
        ("kappa", r.kappa),
        ("stiffness", r.stiffness),
        ("damping_ratio", r.damping_ratio),
        ("travel", r.travel),
        ("friction", r.friction),
        ("mass", r.mass),
        ("gripper_damping_scale", r.gripper_damping_scale),
    ]
}

/// Force-PI baseline (Rust `ForcePi`); `act` returns the gripper delta (rad).
#[pyclass(module = "robot_sim._native", name = "ForcePi")]
pub(crate) struct PyForcePi {
    inner: ForcePi,
}
#[pymethods]
impl PyForcePi {
    #[new]
    #[pyo3(signature = (kp=None, ki=None, stiffness=None, closing_speed=None, threshold=None))]
    fn new(
        kp: Option<f64>,
        ki: Option<f64>,
        stiffness: Option<f64>,
        closing_speed: Option<f64>,
        threshold: Option<f64>,
    ) -> Self {
        let d = ForcePi::default();
        Self {
            inner: ForcePi::new(
                kp.unwrap_or(d.kp),
                ki.unwrap_or(d.ki),
                stiffness.unwrap_or(d.stiffness),
                closing_speed.unwrap_or(d.closing_speed),
                threshold.unwrap_or(d.threshold),
            ),
        }
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn act(&mut self, forces: [f64; 2], goal: f64, dt: f64) -> f64 {
        self.inner.act(forces, goal, dt)
    }
    #[getter]
    fn gains(&self) -> (f64, f64, f64, f64, f64) {
        let p = &self.inner;
        (p.kp, p.ki, p.stiffness, p.closing_speed, p.threshold)
    }
}
