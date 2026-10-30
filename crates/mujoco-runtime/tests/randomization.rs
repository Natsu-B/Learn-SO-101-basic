use mujoco_runtime::{Simulator, presets, randomization::*};
use robotics_core::JointCommand;
#[test]
fn nominal_identity_nonaccumulation_and_finite() {
    for robot in ["so101", "panda", "ur5e"] {
        let mut a = Simulator::from_spec(presets::robot(robot).unwrap()).unwrap();
        let mut b = Simulator::from_spec(presets::robot(robot).unwrap()).unwrap();
        let nominal = a.physical_parameters().unwrap();
        let bounds = a.action_bounds();
        a.reset_seed(42).unwrap();
        let cmd = JointCommand {
            position: a.home_position(),
        };
        for _ in 0..10 {
            assert_eq!(
                a.step_control(&cmd, 1. / 30.).unwrap(),
                b.step_control(&cmd, 1. / 30.).unwrap()
            );
        }
        let c = DomainRandomizationConfig {
            joint_damping_scale: Some(RandomizationRange { min: 0.9, max: 1.1 }),
            frictionloss_scale: Some(RandomizationRange { min: 0.9, max: 1.1 }),
            actuator_gain_scale: Some(RandomizationRange {
                min: 0.95,
                max: 1.05,
            }),
            actuator_force_scale: Some(RandomizationRange {
                min: 0.95,
                max: 1.05,
            }),
            ..Default::default()
        };
        a.configure_randomization(c).unwrap();
        a.reset_seed(1).unwrap();
        let first = a.physical_parameters().unwrap();
        a.reset_seed(2).unwrap();
        assert_ne!(first, a.physical_parameters().unwrap());
        a.reset_seed(1).unwrap();
        assert_eq!(first, a.physical_parameters().unwrap());
        assert_eq!(bounds, a.action_bounds());
        for _ in 0..1000 {
            assert!(a.step_control(&cmd, 1. / 30.).unwrap().is_finite());
        }
        a.configure_randomization(DomainRandomizationConfig::default())
            .unwrap();
        a.reset_seed(3).unwrap();
        assert_eq!(nominal, a.physical_parameters().unwrap());
    }
}

#[test]
fn identified_center_roundtrip_and_randomization_composition() {
    let mut sim = Simulator::new().unwrap();
    let nominal = sim.physical_parameters().unwrap();
    let mut center = sim.parameters().clone();
    center.actuator_gain_scale.fill(1.15);
    center.control_latency_seconds = 0.01;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("identified.json");
    center.save(&path).unwrap();
    let loaded = ModelParameters::load(&path).unwrap();
    assert_eq!(center, loaded);
    sim.set_parameter_center(loaded).unwrap();
    sim.configure_randomization(DomainRandomizationConfig {
        actuator_gain_scale: Some(RandomizationRange { min: 0.9, max: 0.9 }),
        ..Default::default()
    })
    .unwrap();
    for seed in 0..10 {
        sim.reset_seed(seed).unwrap();
        assert_eq!(sim.parameters().actuator_gain_scale[0], 1.15 * 0.9);
        assert_eq!(
            sim.physical_parameters().unwrap()[0][2],
            nominal[0][2] * 1.15 * 0.9
        );
    }
    let mut bad = center;
    bad.robot = "panda".into();
    assert!(sim.set_parameter_center(bad).is_err());
}
