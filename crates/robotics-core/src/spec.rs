use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ActuatorSpec {
    pub name: String,
    pub joint: String,
}

#[derive(Debug, Clone)]
pub struct EndEffectorSpec {
    pub name: String,
    pub site: String,
}

#[derive(Debug, Clone)]
pub struct CameraSpec {
    pub name: String,
    pub source: String,
}

/// Optional scene-owned frame on a model body, when the source lacks a site.
#[derive(Debug, Clone)]
pub struct SiteAttachment {
    pub body: String,
    pub site: String,
    pub position: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct RobotSpec {
    pub name: String,
    pub model_path: PathBuf,
    pub actuators: Vec<ActuatorSpec>,
    pub end_effectors: Vec<EndEffectorSpec>,
    pub default_end_effector: Option<String>,
    pub cameras: Vec<CameraSpec>,
    /// Explicit SI home pose in controlled actuator order.
    pub home_position: Option<Vec<f64>>,
    pub initial_keyframe: Option<String>,
    pub site_attachments: Vec<SiteAttachment>,
}
