//! SO-101 joint API shared by the Rust CLI and Python extension.
pub mod controller;
pub mod error;
pub mod robot;
pub mod simulation;
pub mod state;

pub use error::{Error, Result};
pub use robot::{ACTUATOR_NAMES, RobotBackend};
pub use simulation::Simulator;
pub use state::{JointCommand, JointState};
