use robotics_core::{
    Error, Result, RobotSpec,
    spec::{ActuatorSpec, EndEffectorSpec},
};
use std::path::Path;

pub const SO101_ACTUATORS: [&str; 6] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
];

pub fn so101(directory: impl AsRef<Path>) -> RobotSpec {
    RobotSpec {
        name: "so101".into(),
        model_path: directory.as_ref().join("so101.xml"),
        actuators: SO101_ACTUATORS
            .iter()
            .map(|name| ActuatorSpec {
                name: (*name).into(),
                joint: (*name).into(),
            })
            .collect(),
        default_end_effector: Some("gripper".into()),
        end_effectors: vec![EndEffectorSpec {
            name: "gripper".into(),
            site: "gripperframe".into(),
        }],
        cameras: vec![robotics_core::spec::CameraSpec {
            name: "wrist".into(),
            source: "wrist_cam".into(),
        }],
        home_position: None,
        initial_keyframe: None,
        site_attachments: vec![],
    }
}

pub fn robot(name: &str) -> Result<RobotSpec> {
    match name {
        "so101" => Ok(so101(
            std::env::var_os("SO101_MODEL_DIR").ok_or(Error::MissingModelDirectory)?,
        )),
        "panda" => Ok(panda(
            std::env::var_os("MENAGERIE_DIR")
                .ok_or_else(|| Error::Model("MENAGERIE_DIR is not set".into()))?,
        )),
        "ur5e" => Ok(ur5e(
            std::env::var_os("MENAGERIE_DIR")
                .ok_or_else(|| Error::Model("MENAGERIE_DIR is not set".into()))?,
        )),
        _ => Err(Error::Model(format!("unknown robot {name}"))),
    }
}

pub fn ur5e(menagerie: impl AsRef<Path>) -> RobotSpec {
    let names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ];
    RobotSpec {
        name: "ur5e".into(),
        model_path: menagerie.as_ref().join("universal_robots_ur5e/ur5e.xml"),
        actuators: names
            .into_iter()
            .map(|name| ActuatorSpec {
                name: name.into(),
                joint: format!("{name}_joint"),
            })
            .collect(),
        default_end_effector: Some("gripper".into()),
        end_effectors: vec![EndEffectorSpec {
            name: "gripper".into(),
            site: "attachment_site".into(),
        }],
        cameras: vec![],
        home_position: None,
        initial_keyframe: Some("home".into()),
        site_attachments: vec![],
    }
}

pub fn panda(menagerie: impl AsRef<Path>) -> RobotSpec {
    RobotSpec {
        name: "panda".into(),
        model_path: menagerie.as_ref().join("franka_emika_panda/panda.xml"),
        actuators: (1..=7)
            .map(|i| ActuatorSpec {
                name: format!("actuator{i}"),
                joint: format!("joint{i}"),
            })
            .collect(),
        default_end_effector: Some("gripper".into()),
        end_effectors: vec![EndEffectorSpec {
            name: "gripper".into(),
            site: "panda_hand_site".into(),
        }],
        cameras: vec![],
        home_position: None,
        initial_keyframe: Some("home".into()),
        site_attachments: vec![robotics_core::spec::SiteAttachment {
            body: "hand".into(),
            site: "panda_hand_site".into(),
            position: [0.0; 3],
        }],
    }
}
