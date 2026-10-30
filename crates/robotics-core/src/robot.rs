use crate::{JointCommand, JointState, Result};

/// Boundary for a future real robot backend. Controllers exchange only SI
/// joint state/commands; implementations own their timing and transport.
pub trait RobotBackend {
    fn reset(&mut self) -> Result<()>;
    fn state(&self) -> JointState;
    fn step(&mut self, command: &JointCommand) -> Result<JointState>;
    fn time(&self) -> f64;
    fn end_effector(&self, name: &str) -> Result<[f64; 3]>;
}

/// Hardware-facing I/O only: neither method resets a robot or advances a task.
pub trait RobotIo {
    fn observation(&mut self) -> Result<crate::RobotObservation>;
    fn send_action(&mut self, action: &crate::RobotAction) -> Result<()>;
}
