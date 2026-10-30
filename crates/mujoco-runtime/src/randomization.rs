//! Reset-only parameter configuration. All scales are relative to immutable Menagerie values.
use robotics_core::{Error, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RandomizationRange {
    pub min: f64,
    pub max: f64,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DomainRandomizationConfig {
    pub joint_damping_scale: Option<RandomizationRange>,
    pub frictionloss_scale: Option<RandomizationRange>,
    pub actuator_gain_scale: Option<RandomizationRange>,
    pub actuator_force_scale: Option<RandomizationRange>,
    pub control_latency_seconds: Option<RandomizationRange>,
    pub observation_latency_seconds: Option<RandomizationRange>,
    #[serde(alias = "joint_zero_offset_rad")]
    pub sensor_zero_offset_rad: Option<RandomizationRange>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelParameters {
    pub schema_version: u32,
    pub robot: String,
    pub joint_names: Vec<String>,
    pub joint_damping_scale: Vec<f64>,
    pub frictionloss_scale: Vec<f64>,
    pub actuator_gain_scale: Vec<f64>,
    pub actuator_force_scale: Vec<f64>,
    pub control_latency_seconds: f64,
    pub observation_latency_seconds: f64,
    #[serde(alias = "joint_zero_offset_rad")]
    pub sensor_zero_offset_rad: Vec<f64>,
}
fn invalid() -> Error {
    Error::Input("invalid model/randomization parameters (finite, nonnegative scales; positive gains/force, latency <=1s, |offset|<=0.2rad)".into())
}
impl ModelParameters {
    pub fn nominal(robot: String, joint_names: Vec<String>) -> Self {
        let n = joint_names.len();
        Self {
            schema_version: 1,
            robot,
            joint_names,
            joint_damping_scale: vec![1.; n],
            frictionloss_scale: vec![1.; n],
            actuator_gain_scale: vec![1.; n],
            actuator_force_scale: vec![1.; n],
            control_latency_seconds: 0.,
            observation_latency_seconds: 0.,
            sensor_zero_offset_rad: vec![0.; n],
        }
    }
    pub fn validate(&self, robot: &str, names: &[String]) -> Result<()> {
        if self.schema_version != 1
            || self.robot != robot
            || self.joint_names != names
            || names.is_empty()
            || names.iter().collect::<std::collections::HashSet<_>>().len() != names.len()
        {
            return Err(Error::Input(
                "parameter robot/joint identity mismatch".into(),
            ));
        }
        for (values, lo, hi, strict) in [
            (&self.joint_damping_scale, 0., 10., false),
            (&self.frictionloss_scale, 0., 10., false),
            (&self.actuator_gain_scale, 0., 10., true),
            (&self.actuator_force_scale, 0., 10., true),
            (&self.sensor_zero_offset_rad, -0.2, 0.2, false),
        ] {
            if values.len() != names.len()
                || values
                    .iter()
                    .any(|v| !v.is_finite() || *v < lo || *v > hi || (strict && *v == lo))
            {
                return Err(invalid());
            }
        }
        for t in [
            self.control_latency_seconds,
            self.observation_latency_seconds,
        ] {
            if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                return Err(invalid());
            }
        }
        Ok(())
    }
}
impl DomainRandomizationConfig {
    pub fn validate(&self) -> Result<()> {
        for (r, lo, hi, strict) in [
            (self.joint_damping_scale, 0., 10., false),
            (self.frictionloss_scale, 0., 10., false),
            (self.actuator_gain_scale, 0., 10., true),
            (self.actuator_force_scale, 0., 10., true),
            (self.control_latency_seconds, 0., 1., false),
            (self.observation_latency_seconds, 0., 1., false),
            (self.sensor_zero_offset_rad, -0.2, 0.2, false),
        ] {
            if let Some(r) = r
                && (!r.min.is_finite()
                    || !r.max.is_finite()
                    || r.min > r.max
                    || r.min < lo
                    || r.max > hi
                    || (strict && r.min == lo))
            {
                return Err(invalid());
            }
        }
        Ok(())
    }
    pub fn sample(&self, center: &ModelParameters, seed: u64) -> Result<ModelParameters> {
        self.validate()?;
        let mut rng = seed ^ 0xd1b54a32d192ed03;
        let mut draw = |range: Option<RandomizationRange>, default: f64| {
            let Some(r) = range else {
                return default;
            };
            rng = rng.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            r.min + (r.max - r.min) * ((z >> 11) as f64 / ((1_u64 << 53) as f64))
        };
        let mut p = center.clone();
        for (values, range) in [
            (&mut p.joint_damping_scale, self.joint_damping_scale),
            (&mut p.frictionloss_scale, self.frictionloss_scale),
            (&mut p.actuator_gain_scale, self.actuator_gain_scale),
            (&mut p.actuator_force_scale, self.actuator_force_scale),
        ] {
            for v in values {
                *v *= draw(range, 1.);
            }
        }
        p.control_latency_seconds += draw(self.control_latency_seconds, 0.);
        p.observation_latency_seconds += draw(self.observation_latency_seconds, 0.);
        for v in &mut p.sensor_zero_offset_rad {
            *v += draw(self.sensor_zero_offset_rad, 0.);
        }
        p.validate(&center.robot, &center.joint_names)?;
        Ok(p)
    }
}

impl ModelParameters {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let p: Self = serde_json::from_reader(std::fs::File::open(path)?)
            .map_err(|e| Error::Input(e.to_string()))?;
        p.validate(&p.robot, &p.joint_names)?;
        Ok(p)
    }
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.validate(&self.robot, &self.joint_names)?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        serde_json::to_writer_pretty(file, self).map_err(|e| Error::Input(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sampling_is_deterministic_bounded_and_fixed() {
        let p = ModelParameters::nominal("robot".into(), vec!["j".into()]);
        assert_eq!(
            DomainRandomizationConfig::default().sample(&p, 42).unwrap(),
            p
        );
        let c = DomainRandomizationConfig {
            actuator_gain_scale: Some(RandomizationRange {
                min: 0.95,
                max: 1.05,
            }),
            ..Default::default()
        };
        assert_eq!(c.sample(&p, 1).unwrap(), c.sample(&p, 1).unwrap());
        assert_ne!(c.sample(&p, 1).unwrap(), c.sample(&p, 2).unwrap());
        for seed in 0..100 {
            assert!((0.95..=1.05).contains(&c.sample(&p, seed).unwrap().actuator_gain_scale[0]));
        }
        let c = DomainRandomizationConfig {
            control_latency_seconds: Some(RandomizationRange {
                min: 0.02,
                max: 0.02,
            }),
            ..Default::default()
        };
        assert_eq!(c.sample(&p, 1).unwrap().control_latency_seconds, 0.02);
    }
}
