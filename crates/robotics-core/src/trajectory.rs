//! SI measurement format for SysID, not a learning dataset.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryMetadata {
    pub schema_version: u32,
    pub robot: String,
    pub backend: String,
    pub model_version: String,
    pub joint_names: Vec<String>,
    pub control_period: f64,
    pub physics_timestep: Option<f64>,
    /// Unix epoch seconds of capture creation; samples use elapsed monotonic seconds.
    pub timestamp: f64,
    pub seed: u64,
    pub randomization_parameters: serde_json::Value,
}
impl TrajectoryMetadata {
    pub fn validate(&self) -> Result<()> {
        if ![1, 2].contains(&self.schema_version)
            || self.joint_names.is_empty()
            || self
                .joint_names
                .iter()
                .any(|n| n.is_empty() || n.contains(['\n', '\r']))
            || self
                .joint_names
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != self.joint_names.len()
            || !self.control_period.is_finite()
            || self.control_period <= 0.0
            || !self.timestamp.is_finite()
            || self
                .physics_timestep
                .is_some_and(|t| !t.is_finite() || t <= 0.0)
            || !["mujoco", "real"].contains(&self.backend.as_str())
        {
            return Err(Error::Input("invalid trajectory metadata".into()));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectorySample {
    pub command_time: f64,
    pub applied_time: f64,
    pub measurement_time: f64,
    pub source_measurement_time: f64,
    pub applied_joint_position: Vec<f64>,
    pub commanded_joint_position: Vec<f64>,
    pub measured_joint_position: Vec<f64>,
    pub measured_joint_velocity: Option<Vec<f64>>,
    pub end_effector_position: Option<[f64; 3]>,
}
impl TrajectorySample {
    pub fn validate(&self, n: usize, previous: Option<f64>) -> Result<()> {
        if [
            self.command_time,
            self.applied_time,
            self.measurement_time,
            self.source_measurement_time,
        ]
        .iter()
        .any(|t| !t.is_finite() || *t < 0.0)
            || self.command_time > self.measurement_time + 1e-10
            || self.applied_time > self.measurement_time + 1e-10
            || self.source_measurement_time > self.measurement_time + 1e-10
            || previous.is_some_and(|t| self.measurement_time <= t)
            || [
                &self.commanded_joint_position,
                &self.applied_joint_position,
                &self.measured_joint_position,
            ]
            .iter()
            .any(|v| v.len() != n || v.iter().any(|x| !x.is_finite()))
            || self
                .measured_joint_velocity
                .as_ref()
                .is_some_and(|v| v.len() != n || v.iter().any(|x| !x.is_finite()))
            || self
                .end_effector_position
                .is_some_and(|v| v.iter().any(|x| !x.is_finite()))
        {
            return Err(Error::Input(
                "invalid trajectory dimensions, values or non-increasing timestamp".into(),
            ));
        }
        Ok(())
    }
}
pub trait TrajectoryRecorder {
    fn begin(&mut self, metadata: &TrajectoryMetadata) -> Result<()>;
    fn record(&mut self, sample: &TrajectorySample) -> Result<()>;
    fn finish(&mut self) -> Result<()>;
}
pub struct NullRecorder;
impl TrajectoryRecorder for NullRecorder {
    fn begin(&mut self, _: &TrajectoryMetadata) -> Result<()> {
        Ok(())
    }
    fn record(&mut self, _: &TrajectorySample) -> Result<()> {
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}
