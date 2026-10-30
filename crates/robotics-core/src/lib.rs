//! Backend-independent robot data; no MuJoCo, Python or GUI.
pub mod controller;
pub mod error;
pub mod robot;
pub mod state;
pub use error::{Error, Result};
pub use robot::RobotBackend;
pub use state::{JointCommand, JointState, RobotAction, RobotObservation};
pub mod spec;
pub use spec::RobotSpec;
pub mod task;

pub mod codec;

pub mod telemetry;

pub mod trajectory;
