use mujoco_runtime::{JointCommand, Simulator, presets};

fn load_and_hold(name: &str, dimension: usize) {
    let spec = presets::robot(name).unwrap();
    let expected: Vec<_> = spec.actuators.iter().map(|a| a.name.clone()).collect();
    let mut sim = Simulator::from_spec(spec).unwrap();
    assert_eq!(sim.actuator_names(), expected);
    assert_eq!(sim.actuator_ids().len(), dimension);
    assert_eq!(sim.state().position.len(), dimension);
    assert!(
        sim.end_effector("gripper")
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
    let initial = sim.state();
    let command = JointCommand {
        position: initial.position.clone(),
    };
    for _ in 0..1000 {
        assert!(sim.step(&command).unwrap().is_finite());
    }
    sim.reset().unwrap();
    assert_eq!(sim.state(), initial);
}

#[test]
fn so101() {
    load_and_hold("so101", 6);
}
#[test]
fn panda() {
    load_and_hold("panda", 7);
}

#[test]
fn ur5e() {
    load_and_hold("ur5e", 6);
}

#[test]
fn camera_logical_source_mapping() {
    let spec = presets::robot("so101").unwrap();
    assert_eq!(spec.cameras[0].name, "wrist");
    assert_eq!(spec.cameras[0].source, "wrist_cam");
    Simulator::from_spec(spec).unwrap();
}
