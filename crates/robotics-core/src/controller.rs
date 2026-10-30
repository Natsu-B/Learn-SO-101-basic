use crate::{Error, JointCommand, JointState, Result};

pub trait Controller {
    fn command(&mut self, state: &JointState, time: f64) -> JointCommand;
}

#[derive(Debug, Clone)]
pub struct HomeController {
    home_position: Vec<f64>,
}
impl HomeController {
    pub fn new(home_position: Vec<f64>, bounds: &[[f64; 2]]) -> Result<Self> {
        if home_position.len() != bounds.len() || home_position.is_empty() {
            return Err(Error::Input(
                "home dimension must match nonempty joint bounds".into(),
            ));
        }
        if home_position
            .iter()
            .zip(bounds)
            .any(|(q, [lo, hi])| !q.is_finite() || q < lo || q > hi)
        {
            return Err(Error::Input(
                "home position must be inside joint safe ranges".into(),
            ));
        }
        Ok(Self { home_position })
    }
}
impl Controller for HomeController {
    fn command(&mut self, _state: &JointState, _time: f64) -> JointCommand {
        JointCommand {
            position: self.home_position.clone(),
        }
    }
}

/// A selected joint oscillates around home; all other joints hold home.
#[derive(Debug, Clone)]
pub struct SineController {
    home: HomeController,
    joint_index: usize,
    amplitude: f64,
    frequency: f64,
}
impl SineController {
    pub fn new(
        home_position: Vec<f64>,
        bounds: &[[f64; 2]],
        joint_index: usize,
        amplitude: f64,
        frequency: f64,
    ) -> Result<Self> {
        let home = HomeController::new(home_position, bounds)?;
        if joint_index >= bounds.len()
            || !amplitude.is_finite()
            || amplitude < 0.0
            || !frequency.is_finite()
            || frequency < 0.0
        {
            return Err(Error::Input(
                "invalid sine joint, amplitude or frequency".into(),
            ));
        }
        let q = home.home_position[joint_index];
        let [lo, hi] = bounds[joint_index];
        if q - amplitude < lo || q + amplitude > hi {
            return Err(Error::Input(
                "sine amplitude exceeds safe joint range".into(),
            ));
        }
        Ok(Self {
            home,
            joint_index,
            amplitude,
            frequency,
        })
    }
}
impl Controller for SineController {
    fn command(&mut self, state: &JointState, time: f64) -> JointCommand {
        let mut command = self.home.command(state, time);
        command.position[self.joint_index] +=
            self.amplitude * (std::f64::consts::TAU * self.frequency * time).sin();
        command
    }
}

/// Bounded, finite-duration SI excitation. Index selection happens by joint name
/// in callers; other joints hold home. No backend dependencies.
pub struct SineTrajectory {
    controller: SineController,
    duration: f64,
}
impl SineTrajectory {
    pub fn new(
        home: Vec<f64>,
        bounds: &[[f64; 2]],
        index: usize,
        amplitude: f64,
        frequency: f64,
        duration: f64,
    ) -> Result<Self> {
        if !duration.is_finite() || duration <= 0.0 {
            return Err(Error::Input("duration must be positive and finite".into()));
        }
        Ok(Self {
            controller: SineController::new(home, bounds, index, amplitude, frequency)?,
            duration,
        })
    }
    pub fn sample(&mut self, time: f64) -> Result<JointCommand> {
        if !time.is_finite() || time < 0.0 || time > self.duration {
            return Err(Error::Input("excitation time outside duration".into()));
        }
        Ok(self.controller.command(
            &JointState {
                position: vec![],
                velocity: vec![],
            },
            time,
        ))
    }
}
