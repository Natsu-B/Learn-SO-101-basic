use so101_core::{ACTUATOR_NAMES, Error, JointCommand, Simulator};

#[test]
fn loads_so101_with_correct_runtime_and_timestep() {
    assert_eq!(rusty_mujoco::mj_versionString(), "3.10.0");
    let sim = Simulator::new().unwrap();
    assert_eq!(sim.timestep(), 0.005);
    assert_eq!(sim.time(), 0.0);
    assert_eq!(sim.state().position, [0.0; 6]);
}

#[test]
fn caches_six_named_actuators_in_api_order() {
    let mut sim = Simulator::new().unwrap();
    assert_eq!(
        sim.actuator_names(),
        [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper"
        ]
    );
    let ids = sim.actuator_ids();
    sim.with_mujoco(|model, _| {
        for (name, id) in ACTUATOR_NAMES.iter().zip(ids) {
            let named = model
                .object_id::<rusty_mujoco::obj::Actuator>(name)
                .unwrap();
            assert_eq!(named.index(), id);
        }
    });
}

#[test]
fn runs_1000_steps_with_finite_state() {
    let mut sim = Simulator::new().unwrap();
    for i in 0..1000 {
        let command = JointCommand {
            position: [0.3 * (i as f64 * 0.01).sin(), 0.1, -0.2, 0.1, 0.1, 0.2].to_vec(),
        };
        assert!(sim.step(&command).unwrap().is_finite());
    }
    assert!((sim.time() - 1000.0 * sim.timestep()).abs() < 1e-10);
}

#[test]
fn reset_restores_initial_state_and_time() {
    let mut sim = Simulator::new().unwrap();
    let initial = sim.state();
    let command = JointCommand {
        position: vec![0.2; 6],
    };
    for _ in 0..100 {
        sim.step(&command).unwrap();
    }
    assert_ne!(sim.state(), initial);
    sim.reset().unwrap();
    assert_eq!(sim.state(), initial);
    assert_eq!(sim.time(), 0.0);
}

#[test]
fn reproduces_sequence_after_reset_and_in_new_instance() {
    let mut a = Simulator::new().unwrap();
    let mut b = Simulator::new().unwrap();
    let commands: Vec<_> = (0..200)
        .map(|i| JointCommand {
            position: [0.2 * (i as f64 * 0.04).sin(), 0.1, -0.2, 0.1, 0.0, 0.3].to_vec(),
        })
        .collect();
    let expected: Vec<_> = commands.iter().map(|c| a.step(c).unwrap()).collect();
    for (command, state) in commands.iter().zip(&expected) {
        assert_eq!(b.step(command).unwrap(), *state);
    }
    a.reset().unwrap();
    for (command, state) in commands.iter().zip(&expected) {
        assert_eq!(a.step(command).unwrap(), *state);
    }
}

#[test]
fn invalid_commands_do_not_advance_or_modify_state() {
    let mut sim = Simulator::new().unwrap();
    let initial = sim.state();
    for value in [f64::NAN, f64::INFINITY, 100.0] {
        let command = JointCommand {
            position: [0.0, 0.0, 0.0, value, 0.0, 0.0].to_vec(),
        };
        assert!(matches!(sim.step(&command), Err(Error::Command { .. })));
        assert_eq!(sim.state(), initial);
        assert_eq!(sim.time(), 0.0);
    }
}

#[test]
fn missing_model_is_an_error() {
    let empty = tempfile::tempdir().unwrap();
    assert!(matches!(
        Simulator::from_model_dir(empty.path()),
        Err(Error::Io(_))
    ));
}
