use mujoco_runtime::{Error, JointCommand, Simulator};
use robotics_core::{RobotSpec, spec::ActuatorSpec};

fn fixture() -> (tempfile::TempDir, RobotSpec) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.xml");
    std::fs::write(
        &path,
        r#"<mujoco><compiler angle="radian"/><option timestep="0.005"/><worldbody>
      <body name="a"><joint name="ja" range="-1 1"/><geom type="sphere" size="0.01"/></body>
      <body name="extra" pos="1 0 0"><joint name="jx"/><geom type="sphere" size="0.01"/></body>
      <body name="b" pos="2 0 0"><joint name="jb"/><geom type="sphere" size="0.01"/></body>
      </worldbody><actuator>
      <position name="a" joint="ja" kp="10" ctrlrange="-2 2"/>
      <position name="extra" joint="jx" kp="10"/>
      <position name="b" joint="jb" kp="10" ctrlrange="-0.5 0.5"/>
      </actuator></mujoco>"#,
    )
    .unwrap();
    let spec = RobotSpec {
        name: "fixture".into(),
        model_path: path,
        actuators: vec![
            ActuatorSpec {
                name: "b".into(),
                joint: "jb".into(),
            },
            ActuatorSpec {
                name: "a".into(),
                joint: "ja".into(),
            },
        ],
        default_end_effector: None,
        end_effectors: vec![],
        cameras: vec![],
        home_position: None,
        initial_keyframe: None,
        site_attachments: vec![],
    };
    (directory, spec)
}

#[test]
fn non_contiguous_reordered_subset_and_intersection_limits() {
    let (_directory, spec) = fixture();
    let mut sim = Simulator::from_spec(spec).unwrap();
    assert_eq!(sim.actuator_ids(), [2, 0]);
    assert_eq!(sim.action_bounds(), [[-0.5, 0.5], [-1.0, 1.0]]);
    sim.step(&JointCommand {
        position: vec![0.5, 1.0],
    })
    .unwrap();
    for values in [
        vec![0.6, 0.0],
        vec![0.0, 1.5],
        vec![0.0],
        vec![f64::NAN, 0.0],
        vec![0.0, f64::INFINITY],
    ] {
        let time = sim.time();
        let state = sim.state();
        assert!(sim.step(&JointCommand { position: values }).is_err());
        assert_eq!(sim.time(), time);
        assert_eq!(sim.state(), state);
    }
}

#[test]
fn unbounded_actuator_and_joint_accept_only_finite_targets() {
    let (_directory, mut spec) = fixture();
    spec.actuators = vec![ActuatorSpec {
        name: "extra".into(),
        joint: "jx".into(),
    }];
    let mut sim = Simulator::from_spec(spec).unwrap();
    assert_eq!(sim.action_bounds(), [[f64::NEG_INFINITY, f64::INFINITY]]);
    sim.step(&JointCommand {
        position: vec![20.0],
    })
    .unwrap();
    assert!(matches!(
        sim.step(&JointCommand {
            position: vec![f64::NAN]
        }),
        Err(Error::Command { .. })
    ));
}

#[test]
fn transmission_mismatch_is_rejected() {
    let (_directory, mut spec) = fixture();
    spec.actuators[0].joint = "ja".into();
    assert!(matches!(Simulator::from_spec(spec), Err(Error::Model(_))));
}

#[test]
fn so101_wrist_roll_joint_limit_is_stricter_than_ctrlrange() {
    let mut sim = Simulator::new().unwrap();
    let mut values = vec![0.0; 6];
    values[4] = 2.8;
    assert!(sim.step(&JointCommand { position: values }).is_err());
}
