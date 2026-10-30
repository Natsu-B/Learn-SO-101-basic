use mujoco_runtime::{JointCommand, Simulator};

#[test]
fn thirty_hz_control_does_not_drift_against_200_hz_physics() {
    let mut sim = Simulator::new().unwrap();
    assert_eq!(sim.timestep(), 0.005);
    let command = JointCommand {
        position: sim.state().position,
    };
    for i in 1..=6000 {
        assert!(sim.step_control(&command, 1.0 / 30.0).unwrap().is_finite());
        let ideal = i as f64 / 30.0;
        assert!(sim.time() >= ideal - 1e-8);
        assert!(sim.time() < ideal + sim.timestep() + 1e-8);
    }
    assert!((sim.time() - 200.0).abs() < 1e-8);
}

#[test]
fn physics_control_reset_and_subphysics_period() {
    let mut sim = Simulator::new().unwrap();
    let command = JointCommand {
        position: sim.state().position,
    };
    sim.step_physics(&command).unwrap();
    assert_eq!(sim.time(), 0.005);
    sim.step_control(&command, 1.0 / 30.0).unwrap();
    assert!((sim.time() - 0.04).abs() < 1e-12);
    sim.reset().unwrap();
    for _ in 0..10 {
        sim.step_control(&command, 0.001).unwrap();
    }
    assert!((sim.time() - 0.01).abs() < 1e-12);
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(sim.step_control(&command, invalid).is_err());
        assert!((sim.time() - 0.01).abs() < 1e-12);
    }
}
