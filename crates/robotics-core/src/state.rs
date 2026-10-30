/// Angles in radians and angular velocities in radians/second.
/// The gripper is also a hinge angle, not a jaw distance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JointState {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
}

impl JointState {
    pub fn is_finite(&self) -> bool {
        self.position
            .iter()
            .chain(&self.velocity)
            .all(|v| v.is_finite())
    }
}

/// Targets for the model's existing position actuators, in radians.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JointCommand {
    pub position: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobotObservation {
    pub joint_position: Vec<f32>,
    pub joint_velocity: Vec<f32>,
}
impl From<JointState> for RobotObservation {
    fn from(state: JointState) -> Self {
        Self {
            joint_position: state.position.into_iter().map(|v| v as f32).collect(),
            joint_velocity: state.velocity.into_iter().map(|v| v as f32).collect(),
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum RobotAction {
    JointPosition(Vec<f32>),
}
impl From<RobotAction> for JointCommand {
    fn from(action: RobotAction) -> Self {
        match action {
            RobotAction::JointPosition(v) => Self {
                position: v.into_iter().map(f64::from).collect(),
            },
        }
    }
}
