//! MuJoCo ownership, model loading and cached joint mapping.
use crate::{
    presets,
    randomization::{DomainRandomizationConfig, ModelParameters},
};
use robotics_core::{Error, JointCommand, JointState, Result, RobotBackend, RobotSpec};
use rusty_mujoco::{self as mj, ObjectId, joint, obj};
use std::{collections::VecDeque, io::Write, path::Path};

struct JointMapping {
    actuator: ObjectId<obj::Actuator>,
    joint: ObjectId<joint::Hinge>,
    range: [f64; 2],
}

#[derive(Debug, Clone)]
pub struct ObservationSnapshot {
    pub timestamp: f64,
    pub joint_state: JointState,
    pub end_effector_position: Option<[f64; 3]>,
}
pub struct Simulator {
    // mjData must be destroyed before its corresponding model.
    pub(crate) data: mj::mjData,
    pub(crate) fk_data: std::cell::RefCell<mj::mjData>,
    pub(crate) model: mj::mjModel,
    joints: Vec<JointMapping>,
    spec: RobotSpec,
    sites: Vec<(String, ObjectId<obj::Site>)>,
    healthy: bool,
    command_queue: VecDeque<(f64, f64, JointCommand)>,
    history: VecDeque<ObservationSnapshot>,
    applied_command: JointCommand,
    command_time: f64,
    applied_time: f64,
    nominal_physics: Vec<[f64; 7]>,
    parameter_center: ModelParameters,
    parameters: ModelParameters,
    randomization: DomainRandomizationConfig,
    home_position: Vec<f64>,
    physics_steps: u64,
    control_deadline: f64,
    deadline_compensation: f64,
}

impl Simulator {
    /// Load the repository scene using SO101_MODEL_DIR from the Nix devShell.
    pub fn new() -> Result<Self> {
        let directory = std::env::var_os("SO101_MODEL_DIR").ok_or(Error::MissingModelDirectory)?;
        Self::from_model_dir(directory)
    }

    /// Directory containing the unchanged Menagerie so101.xml and assets/.
    pub fn from_model_dir(directory: impl AsRef<Path>) -> Result<Self> {
        Self::from_spec(presets::so101(directory))
    }

    pub fn from_spec(spec: RobotSpec) -> Result<Self> {
        Self::from_spec_with_scene(spec, include_str!("../../../scenes/robot.xml"))
    }

    /// Load a robot into a caller-provided scene template. The template must
    /// contain `@MODEL_XML@` and `@MODEL_ASSETS@` placeholders (see scenes/).
    pub fn from_spec_with_scene(spec: RobotSpec, scene_template: &str) -> Result<Self> {
        if !scene_template.contains("@MODEL_XML@") || !scene_template.contains("@MODEL_ASSETS@") {
            return Err(Error::Model(
                "scene template needs @MODEL_XML@ and @MODEL_ASSETS@".into(),
            ));
        }
        let version = mj::mj_versionString();
        if version != "3.10.0" {
            return Err(Error::Version(version));
        }
        let xml_path = spec.model_path.canonicalize()?;
        let directory = xml_path
            .parent()
            .ok_or_else(|| Error::Model("model has no parent".into()))?;
        std::fs::metadata(&xml_path)?;
        let mut adapted = tempfile::NamedTempFile::new()?;
        let include_path = if spec.site_attachments.is_empty() {
            xml_path.clone()
        } else {
            let mut source = std::fs::read_to_string(&xml_path)?;
            for attachment in &spec.site_attachments {
                let document =
                    roxmltree::Document::parse(&source).map_err(|e| Error::Model(e.to_string()))?;
                let body = document
                    .descendants()
                    .find(|n| {
                        n.has_tag_name("body")
                            && n.attribute("name") == Some(attachment.body.as_str())
                    })
                    .ok_or_else(|| {
                        Error::Model(format!("missing adapter body {}", attachment.body))
                    })?;
                let end = body.range().end;
                let closing = source[..end]
                    .rfind("</body>")
                    .ok_or_else(|| Error::Model("invalid adapter body".into()))?;
                let [x, y, z] = attachment.position;
                let site = xml_attribute(Path::new(&attachment.site))?;
                source.insert_str(
                    closing,
                    &format!("<site name=\"{site}\" pos=\"{x} {y} {z}\" size=\"0.003\"/>"),
                );
            }
            adapted.write_all(source.as_bytes())?;
            adapted.path().to_owned()
        };
        let xml = scene_template
            .replace("@MODEL_XML@", &xml_attribute(&include_path)?)
            .replace("@MODEL_ASSETS@", &xml_attribute(&directory.join("assets"))?);
        // mj_loadXML avoids rusty_mujoco 0.4.0's broken mjSpec Drop path.
        // The temporary scene is needed only during compilation.
        let mut scene = tempfile::NamedTempFile::new()?;
        scene.write_all(xml.as_bytes())?;
        let filename = scene
            .path()
            .to_str()
            .ok_or_else(|| Error::Model("non-UTF8 scene path".into()))?;
        let model = mj::mj_loadXML(filename).map_err(|e| Error::Model(e.to_string()))?;
        if !model.opt().timestep().is_finite() || model.opt().timestep() <= 0.0 {
            return Err(Error::Model("invalid model timestep".into()));
        }
        let mut mappings = Vec::with_capacity(spec.actuators.len());
        for actuator_spec in &spec.actuators {
            let name = &actuator_spec.name;
            let actuator = model
                .object_id::<obj::Actuator>(name)
                .ok_or_else(|| Error::Model(format!("missing actuator {name}")))?;
            let joint = model
                .object_id::<joint::Hinge>(&actuator_spec.joint)
                .ok_or_else(|| Error::Model(format!("missing hinge joint {name}")))?;
            if crate::model_access::actuator_joint(&model, actuator) != Some(joint.index()) {
                return Err(Error::Model(format!(
                    "actuator {name} must transmit to {}",
                    actuator_spec.joint
                )));
            }
            let gear = model.actuator_gear(actuator);
            let gain = model.actuator_gainprm(actuator);
            let bias = model.actuator_biasprm(actuator);
            if gear != [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                || model.actuator_gaintype(actuator) != mj::mjtGain::FIXED
                || model.actuator_biastype(actuator) != mj::mjtBias::AFFINE
                || gain[0] <= 0.0
                || bias[0] != 0.0
                || bias[1] != -gain[0]
            {
                return Err(Error::Model(format!(
                    "actuator {name} is not a supported unit-gear joint position servo"
                )));
            }
            let ctrl = model
                .actuator_ctrlrange(actuator)
                .map(|r| [r.start, r.end])
                .unwrap_or([f64::NEG_INFINITY, f64::INFINITY]);
            let joint_range = model
                .jnt_range(joint)
                .map(|r| [r.start, r.end])
                .unwrap_or([f64::NEG_INFINITY, f64::INFINITY]);
            let range = [ctrl[0].max(joint_range[0]), ctrl[1].min(joint_range[1])];
            if range[0] > range[1] {
                return Err(Error::Model(format!("empty safe range for {name}")));
            }
            mappings.push(JointMapping {
                actuator,
                joint,
                range,
            });
        }
        let joints = mappings;
        let mut sites = Vec::new();
        for effector in &spec.end_effectors {
            let id = model
                .object_id::<obj::Site>(&effector.site)
                .ok_or_else(|| {
                    Error::Model(format!("missing end-effector site {}", effector.site))
                })?;
            sites.push((effector.name.clone(), id));
        }
        if let Some(name) = &spec.default_end_effector
            && !sites.iter().any(|(n, _)| n == name)
        {
            return Err(Error::Model(format!("unknown default end effector {name}")));
        }
        for camera in &spec.cameras {
            model
                .object_id::<obj::Camera>(&camera.source)
                .ok_or_else(|| {
                    Error::Model(format!(
                        "missing camera {} for {}",
                        camera.source, camera.name
                    ))
                })?;
        }
        let data = mj::mj_makeData(&model);
        let fk_data = std::cell::RefCell::new(mj::mj_makeData(&model));
        let nominal_physics = joints
            .iter()
            .map(|j| crate::model_access::parameters(&model, j.actuator))
            .collect::<Result<Vec<_>>>()?;
        let parameter_center = ModelParameters::nominal(
            spec.name.clone(),
            spec.actuators.iter().map(|a| a.joint.clone()).collect(),
        );
        let mut simulator = Self {
            command_queue: VecDeque::new(),
            history: VecDeque::new(),
            applied_command: JointCommand::default(),
            command_time: 0.0,
            applied_time: 0.0,
            nominal_physics,
            parameters: parameter_center.clone(),
            parameter_center,
            randomization: DomainRandomizationConfig::default(),
            fk_data,
            data,
            model,
            joints,
            healthy: true,
            home_position: Vec::new(),
            physics_steps: 0,
            control_deadline: 0.0,
            deadline_compensation: 0.0,
            spec,
            sites,
        };
        simulator.reset()?;
        let mut defaults = mj::mj_makeData(&simulator.model);
        mj::mj_resetData(&simulator.model, &mut defaults);
        let qpos0: Vec<_> = simulator
            .joints
            .iter()
            .map(|j| defaults.qpos(j.joint, &simulator.model)[0])
            .collect();
        let home = if let Some(home) = &simulator.spec.home_position {
            home.clone()
        } else if simulator
            .validate_command(&JointCommand {
                position: qpos0.clone(),
            })
            .is_ok()
        {
            qpos0
        } else {
            simulator.state().position
        };
        simulator.validate_command(&JointCommand {
            position: home.clone(),
        })?;
        simulator.home_position = home;
        Ok(simulator)
    }

    pub fn configure_randomization(&mut self, config: DomainRandomizationConfig) -> Result<()> {
        config.validate()?;
        self.randomization = config;
        Ok(())
    }
    pub fn set_parameter_center(&mut self, parameters: ModelParameters) -> Result<()> {
        parameters.validate(&self.spec.name, &self.parameter_center.joint_names)?;
        self.parameter_center = parameters;
        Ok(())
    }
    pub fn parameters(&self) -> &ModelParameters {
        &self.parameters
    }
    pub fn physical_parameters(&self) -> Result<Vec<[f64; 7]>> {
        self.joints
            .iter()
            .map(|j| crate::model_access::parameters(&self.model, j.actuator))
            .collect()
    }
    pub fn reset_seed(&mut self, seed: u64) -> Result<()> {
        let parameters = self.randomization.sample(&self.parameter_center, seed)?;
        let mut physical = Vec::with_capacity(self.joints.len());
        for (i, j) in self.joints.iter().enumerate() {
            if self.model.actuator_forcerange(j.actuator).is_none()
                && parameters.actuator_force_scale[i] != 1.0
            {
                return Err(Error::Input(
                    "force scale requires a limited actuator".into(),
                ));
            }
            let mut p = self.nominal_physics[i];
            p[0] *= parameters.joint_damping_scale[i];
            p[1] *= parameters.frictionloss_scale[i];
            // Position servo kp requires matching -kp bias; kv remains nominal.
            p[2] *= parameters.actuator_gain_scale[i];
            p[3] *= parameters.actuator_gain_scale[i];
            p[5] *= parameters.actuator_force_scale[i];
            p[6] *= parameters.actuator_force_scale[i];
            if p.iter().any(|v| !v.is_finite()) {
                return Err(Error::Input("physical parameter overflow".into()));
            }
            physical.push(p);
        }
        for (j, p) in self.joints.iter().zip(&physical) {
            crate::model_access::set_parameters(&mut self.model, j.actuator, p)?;
        }
        self.parameters = parameters;
        self.reset_data()
    }

    /// Restore model qpos0, velocity, time and solver state. Set position targets
    /// to qpos0 before forwarding the model.
    pub fn reset(&mut self) -> Result<()> {
        self.reset_seed(0)
    }
    fn reset_data(&mut self) -> Result<()> {
        self.command_queue.clear();
        self.command_time = 0.0;
        self.applied_time = 0.0;
        self.history.clear();
        self.physics_steps = 0;
        self.control_deadline = 0.0;
        self.deadline_compensation = 0.0;
        if let Some(name) = &self.spec.initial_keyframe {
            let key = self
                .model
                .object_id::<obj::Key>(name)
                .ok_or_else(|| Error::Model(format!("missing keyframe {name}")))?;
            mj::mj_resetDataKeyframe(&self.model, &mut self.data, Some(key));
        } else {
            mj::mj_resetData(&self.model, &mut self.data);
        }
        for mapping in &self.joints {
            self.data.set_ctrl(
                mapping.actuator,
                self.data.qpos(mapping.joint, &self.model)[0],
            );
        }
        mj::mj_forward(&self.model, &mut self.data);
        self.healthy = self.state().is_finite();
        self.applied_command = JointCommand {
            position: self.state().position,
        };
        self.capture_observation()?;
        if self.healthy {
            Ok(())
        } else {
            Err(Error::Unstable)
        }
    }

    pub fn state(&self) -> JointState {
        JointState {
            position: self
                .joints
                .iter()
                .map(|j| self.data.qpos(j.joint, &self.model)[0])
                .collect(),
            velocity: self
                .joints
                .iter()
                .map(|j| self.data.qvel(j.joint, &self.model)[0])
                .collect(),
        }
    }

    /// Validate all targets before modifying data, then run exactly one mj_step.
    /// Out-of-range targets are rejected; the model enforces physical limits.
    pub fn validate_command(&self, command: &JointCommand) -> Result<()> {
        if command.position.len() != self.joints.len() {
            return Err(Error::Dimension {
                expected: self.joints.len(),
                actual: command.position.len(),
            });
        }
        if !self.healthy {
            return Err(Error::Unstable);
        }
        for (i, mapping) in self.joints.iter().enumerate() {
            let value = command.position[i];
            let [min, max] = mapping.range;
            if !value.is_finite() || value < min || value > max {
                return Err(Error::Command {
                    joint: self.spec.actuators[i].name.clone(),
                    value,
                    min,
                    max,
                });
            }
        }
        Ok(())
    }

    /// Compatibility alias: exactly one physics step.
    pub fn step(&mut self, command: &JointCommand) -> Result<JointState> {
        self.step_physics(command)
    }
    pub fn step_physics(&mut self, command: &JointCommand) -> Result<JointState> {
        self.enqueue_command(command)?;
        let state = self.advance_physics()?;
        // Mixing explicit physics stepping with control starts a new schedule.
        self.control_deadline = self.physics_steps as f64 * self.timestep();
        self.deadline_compensation = 0.0;
        Ok(state)
    }

    /// Advance to an absolute control deadline, holding the target. The deadline
    /// uses compensated addition, while physics scheduling uses integer ticks.
    /// Observed time overshoots the ideal deadline by less than one physics tick.
    pub fn step_control(&mut self, command: &JointCommand, period: f64) -> Result<JointState> {
        self.validate_command(command)?;
        if !period.is_finite() || period <= 0.0 || !(self.control_deadline + period).is_finite() {
            return Err(Error::Input(
                "control period must be finite and positive".into(),
            ));
        }
        let increment = period - self.deadline_compensation;
        let next = self.control_deadline + increment;
        self.deadline_compensation = (next - self.control_deadline) - increment;
        self.control_deadline = next;
        self.enqueue_command(command)?;
        while self.physics_steps as f64 * self.timestep() + self.timestep() * 1e-9
            < self.control_deadline
        {
            self.advance_physics()?;
        }
        Ok(self.state())
    }

    pub fn step_action(
        &mut self,
        action: &robotics_core::RobotAction,
        period: f64,
    ) -> Result<robotics_core::RobotObservation> {
        Ok(self.step_control(&action.clone().into(), period)?.into())
    }

    fn advance_physics(&mut self) -> Result<JointState> {
        self.apply_due_commands();
        let expected_time = self.time() + self.timestep();
        mj::mj_step(&self.model, &mut self.data);
        self.physics_steps += 1;
        // mj_step integrates qpos after computing frames. Refresh only kinematics
        // so the observed end-effector belongs to the returned joint state.
        mj::mj_kinematics(&self.model, &mut self.data);
        let state = self.state();
        self.capture_observation()?;
        self.healthy = state.is_finite()
            && self.time().is_finite()
            && (self.time() - expected_time).abs() <= 1e-10 * expected_time.abs().max(1.0);
        if self.healthy {
            Ok(state)
        } else {
            Err(Error::Unstable)
        }
    }

    fn enqueue_command(&mut self, command: &JointCommand) -> Result<()> {
        self.validate_command(command)?;
        let due =
            self.physics_steps as f64 * self.timestep() + self.parameters.control_latency_seconds;
        // Coalesce commands submitted at the same physics instant (sub-tick controls).
        if self.command_queue.back().is_some_and(|(t, _, _)| *t == due) {
            self.command_queue.pop_back();
        }
        self.command_time = self.time();
        self.command_queue
            .push_back((due, self.command_time, command.clone()));
        self.apply_due_commands();
        Ok(())
    }
    fn apply_due_commands(&mut self) {
        let now = self.physics_steps as f64 * self.timestep();
        while self
            .command_queue
            .front()
            .is_some_and(|(t, _, _)| *t <= now + self.timestep() * 1e-9)
        {
            let (_, _, command) = self.command_queue.pop_front().expect("checked nonempty");
            self.applied_time = self.time();
            for (mapping, value) in self.joints.iter().zip(&command.position) {
                self.data.set_ctrl(mapping.actuator, *value);
            }
            self.applied_command = command;
        }
    }
    fn capture_observation(&mut self) -> Result<()> {
        self.history.push_back(ObservationSnapshot {
            timestamp: self.time(),
            joint_state: self.state(),
            end_effector_position: self
                .spec
                .default_end_effector
                .as_deref()
                .map(|n| self.end_effector(n))
                .transpose()?,
        });
        let target = self.time() - self.parameters.observation_latency_seconds;
        while self.history.len() > 1 && self.history[1].timestamp <= target + self.timestep() * 1e-9
        {
            self.history.pop_front();
        }
        Ok(())
    }
    /// Nearest previous physics sample; warmup uses the initial sample.
    /// Zero offset models encoder bias only, not a change in physical geometry.
    pub fn observed(&self) -> ObservationSnapshot {
        let mut observation = self
            .history
            .front()
            .expect("reset initializes history")
            .clone();
        for (q, offset) in observation
            .joint_state
            .position
            .iter_mut()
            .zip(&self.parameters.sensor_zero_offset_rad)
        {
            *q += offset;
        }
        observation
    }
    pub fn applied_command(&self) -> &JointCommand {
        &self.applied_command
    }
    pub fn buffer_sizes(&self) -> (usize, usize) {
        (self.command_queue.len(), self.history.len())
    }

    pub fn trajectory_metadata(
        &self,
        control_period: f64,
        seed: u64,
    ) -> Result<robotics_core::trajectory::TrajectoryMetadata> {
        let metadata = robotics_core::trajectory::TrajectoryMetadata {
            schema_version: 2,
            robot: self.spec.name.clone(),
            backend: "mujoco".into(),
            model_version: format!(
                "MuJoCo {} / {} / Menagerie c96a32d28fb5da84da38c1da4d749e7a13212855",
                mj::mj_versionString(),
                self.spec.model_path.display()
            ),
            joint_names: self
                .spec
                .actuators
                .iter()
                .map(|a| a.joint.clone())
                .collect(),
            control_period,
            physics_timestep: Some(self.timestep()),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| Error::Input(e.to_string()))?
                .as_secs_f64(),
            seed,
            randomization_parameters: serde_json::to_value(&self.parameters)
                .map_err(|e| Error::Input(e.to_string()))?,
        };
        metadata.validate()?;
        Ok(metadata)
    }
    pub fn trajectory_sample(
        &self,
        command: &JointCommand,
    ) -> Result<robotics_core::trajectory::TrajectorySample> {
        self.validate_command(command)?;
        let observation = self.observed();
        let state = observation.joint_state;
        Ok(robotics_core::trajectory::TrajectorySample {
            command_time: self.command_time,
            applied_time: self.applied_time,
            measurement_time: self.time(),
            source_measurement_time: observation.timestamp,
            applied_joint_position: self.applied_command.position.clone(),
            commanded_joint_position: command.position.clone(),
            measured_joint_position: state.position,
            measured_joint_velocity: Some(state.velocity),
            end_effector_position: observation.end_effector_position,
        })
    }
    pub fn time(&self) -> f64 {
        self.data.time()
    }
    pub fn timestep(&self) -> f64 {
        self.model.opt().timestep()
    }
    pub fn actuator_names(&self) -> Vec<String> {
        self.spec.actuators.iter().map(|a| a.name.clone()).collect()
    }
    pub fn actuator_ids(&self) -> Vec<usize> {
        self.joints.iter().map(|j| j.actuator.index()).collect()
    }
    pub(crate) fn joint_ids(&self) -> Vec<ObjectId<joint::Hinge>> {
        self.joints.iter().map(|j| j.joint).collect()
    }
    pub fn home_position(&self) -> Vec<f64> {
        self.home_position.clone()
    }

    pub fn action_bounds(&self) -> Vec<[f64; 2]> {
        self.joints.iter().map(|j| j.range).collect()
    }

    pub fn default_end_effector(&self) -> Result<&str> {
        self.spec
            .default_end_effector
            .as_deref()
            .ok_or_else(|| Error::Model("robot has no default end effector".into()))
    }
    pub fn spec(&self) -> &RobotSpec {
        &self.spec
    }
    pub fn forward_kinematics(&self, position: &[f64], name: &str) -> Result<[f64; 3]> {
        self.validate_command(&JointCommand {
            position: position.to_vec(),
        })?;
        let mut data = self.fk_data.borrow_mut();
        if let Some(name) = &self.spec.initial_keyframe {
            let key = self
                .model
                .object_id::<obj::Key>(name)
                .ok_or_else(|| Error::Model("missing FK keyframe".into()))?;
            mj::mj_resetDataKeyframe(&self.model, &mut data, Some(key));
        } else {
            mj::mj_resetData(&self.model, &mut data);
        }
        for (mapping, q) in self.joints.iter().zip(position) {
            data.set_qpos(mapping.joint, [*q], &self.model);
        }
        mj::mj_forward(&self.model, &mut data);
        self.site_position(&data, name)
    }
    pub fn end_effector(&self, name: &str) -> Result<[f64; 3]> {
        self.site_position(&self.data, name)
    }
    fn site_position(&self, data: &mj::mjData, name: &str) -> Result<[f64; 3]> {
        let (_, id) = self
            .sites
            .iter()
            .find(|(n, _)| n == name)
            .ok_or_else(|| Error::Model(format!("unknown end effector {name}")))?;
        let offset = id.index() * 3;
        // SAFETY: both callers use data allocated for this owned model; none is replaced.
        let positions = unsafe { data.site_xpos(&self.model) };
        Ok([
            positions[offset],
            positions[offset + 1],
            positions[offset + 2],
        ])
    }
    /// Update a MuJoCo visualization scene using this simulator's matched data.
    /// Context/window creation remains the CLI's responsibility.
    pub fn update_scene(
        &mut self,
        options: &mj::mjvOption,
        camera: &mut mj::mjvCamera,
        scene: &mut mj::mjvScene,
    ) {
        mj::mjv_updateScene(
            &self.model,
            &mut self.data,
            options,
            None,
            camera,
            mj::mjtCatBit::ALL,
            scene,
        );
    }
    /// Advanced Rust-only visualization integration. Core has no GUI dependency;
    /// this access is never exposed through Python.
    pub fn with_mujoco<T>(&mut self, f: impl FnOnce(&mj::mjModel, &mj::mjData) -> T) -> T {
        f(&self.model, &self.data)
    }
}

impl robotics_core::robot::RobotIo for Simulator {
    fn observation(&mut self) -> Result<robotics_core::RobotObservation> {
        Ok(self.observed().joint_state.into())
    }
    fn send_action(&mut self, action: &robotics_core::RobotAction) -> Result<()> {
        let command: JointCommand = action.clone().into();
        self.validate_command(&command)?;
        self.enqueue_command(&command)
    }
}
impl robotics_core::task::ReachGeometry for Simulator {
    fn default_end_effector(&self) -> Result<&str> {
        Simulator::default_end_effector(self)
    }
    fn end_effector(&self, name: &str) -> Result<[f64; 3]> {
        Simulator::end_effector(self, name)
    }
    fn action_bounds(&self) -> Vec<[f64; 2]> {
        Simulator::action_bounds(self)
    }
    fn forward_kinematics(&self, position: &[f64], name: &str) -> Result<[f64; 3]> {
        Simulator::forward_kinematics(self, position, name)
    }
}

impl RobotBackend for Simulator {
    fn end_effector(&self, name: &str) -> Result<[f64; 3]> {
        Simulator::end_effector(self, name)
    }
    fn reset(&mut self) -> Result<()> {
        Simulator::reset(self)
    }
    fn state(&self) -> JointState {
        Simulator::state(self)
    }
    fn step(&mut self, command: &JointCommand) -> Result<JointState> {
        Simulator::step(self, command)
    }
    fn time(&self) -> f64 {
        Simulator::time(self)
    }
}

fn xml_attribute(path: &Path) -> Result<String> {
    let value = path
        .to_str()
        .ok_or_else(|| Error::Model("non-UTF8 model path".into()))?;
    if value.contains('\0') {
        return Err(Error::Model("NUL in model path".into()));
    }
    Ok(value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;"))
}
