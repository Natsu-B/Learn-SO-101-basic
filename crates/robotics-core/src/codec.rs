//! Explicit boundary between external policy/hardware units and internal SI.
use crate::{Error, Result, RobotAction};

pub trait JointCodec {
    fn decode(&self, external: &[f32]) -> Result<RobotAction>;
    fn encode(&self, internal: &RobotAction) -> Result<Vec<f32>>;
}

/// Per-joint affine map: radians = external * scale + offset.
/// Degrees use scale=PI/180; normalized/gripper-percent maps use calibrated
/// ranges. This is a conversion, not limit validation or a hardware driver.
pub struct AffineJointCodec {
    scale: Vec<f32>,
    offset: Vec<f32>,
}
impl AffineJointCodec {
    pub fn new(scale: Vec<f32>, offset: Vec<f32>) -> Result<Self> {
        if scale.len() != offset.len()
            || scale.is_empty()
            || scale.iter().any(|s| !s.is_finite() || *s == 0.0)
            || offset.iter().any(|s| !s.is_finite())
        {
            return Err(Error::Input(
                "codec requires matching finite nonzero scales and finite offsets".into(),
            ));
        }
        Ok(Self { scale, offset })
    }
    fn validate(&self, values: &[f32]) -> Result<()> {
        if values.len() != self.scale.len() {
            return Err(Error::Dimension {
                expected: self.scale.len(),
                actual: values.len(),
            });
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Input("codec values must be finite".into()));
        }
        Ok(())
    }
}
impl JointCodec for AffineJointCodec {
    fn decode(&self, external: &[f32]) -> Result<RobotAction> {
        self.validate(external)?;
        let values: Vec<_> = external
            .iter()
            .zip(&self.scale)
            .zip(&self.offset)
            .map(|((&v, &s), &o)| v * s + o)
            .collect();
        self.validate(&values)?;
        Ok(RobotAction::JointPosition(values))
    }
    fn encode(&self, internal: &RobotAction) -> Result<Vec<f32>> {
        let RobotAction::JointPosition(values) = internal;
        self.validate(values)?;
        let result: Vec<_> = values
            .iter()
            .zip(&self.scale)
            .zip(&self.offset)
            .map(|((&v, &s), &o)| (v - o) / s)
            .collect();
        self.validate(&result)?;
        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn degree_and_gripper_percent_roundtrip_and_validation() {
        let codec =
            AffineJointCodec::new(vec![std::f32::consts::PI / 180.0, 0.01], vec![0.0, -0.5])
                .unwrap();
        let action = codec.decode(&[90.0, 100.0]).unwrap();
        let RobotAction::JointPosition(values) = &action;
        assert!((values[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(values[1], 0.5);
        assert_eq!(codec.encode(&action).unwrap(), [90.0, 100.0]);
        assert!(codec.decode(&[0.0]).is_err());
        assert!(codec.decode(&[f32::NAN, 0.0]).is_err());
        assert!(AffineJointCodec::new(vec![0.0], vec![0.0]).is_err());
    }
}
