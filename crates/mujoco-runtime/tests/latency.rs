use mujoco_runtime::Simulator;
use robotics_core::JointCommand;
#[test]
fn command_delay_seconds_observation_history_and_bounded_queues() {
    for delay in [0., 0.01, 0.02, 0.04] {
        let mut sim = Simulator::new().unwrap();
        let mut p = sim.parameters().clone();
        p.control_latency_seconds = delay;
        p.observation_latency_seconds = 0.03;
        sim.set_parameter_center(p).unwrap();
        sim.reset_seed(1).unwrap();
        let home = sim.state();
        let mut command = JointCommand {
            position: home.position.clone(),
        };
        command.position[0] += 0.2;
        let ticks = (delay / sim.timestep()).round() as usize;
        for _ in 0..ticks {
            sim.step_physics(&command).unwrap();
            assert_eq!(sim.applied_command().position, home.position);
        }
        sim.step_physics(&command).unwrap();
        assert_eq!(sim.applied_command(), &command);
        for _ in 0..2000 {
            sim.step_control(&command, 1. / 30.).unwrap();
        }
        let obs = sim.observed();
        assert!((sim.time() - obs.timestamp - 0.03).abs() < sim.timestep() + 1e-8);
        let (queue, history) = sim.buffer_sizes();
        assert!(queue <= 10);
        assert!(history <= 8);
        sim.reset_seed(1).unwrap();
        assert_eq!(sim.observed().timestamp, 0.);
        sim.step_physics(&command).unwrap();
        assert_eq!(sim.observed().joint_state, home);
    }
}
#[test]
fn sensor_offset_does_not_change_true_state_or_safe_ranges() {
    let mut sim = Simulator::new().unwrap();
    let before = sim.state();
    let bounds = sim.action_bounds();
    let mut p = sim.parameters().clone();
    p.sensor_zero_offset_rad[0] = 0.01;
    sim.set_parameter_center(p).unwrap();
    sim.reset_seed(1).unwrap();
    assert_eq!(sim.state(), before);
    assert_eq!(sim.action_bounds(), bounds);
    assert_eq!(
        sim.observed().joint_state.position[0],
        before.position[0] + 0.01
    );
}

#[test]
fn trajectory_v2_separates_requested_applied_and_acquisition_times() {
    let mut sim = Simulator::new().unwrap();
    let mut p = sim.parameters().clone();
    p.control_latency_seconds = 0.04;
    p.observation_latency_seconds = 0.03;
    sim.set_parameter_center(p).unwrap();
    sim.reset().unwrap();
    let home = sim.home_position();
    let mut command = JointCommand {
        position: home.clone(),
    };
    command.position[0] += 0.2;
    sim.step_control(&command, 0.01).unwrap();
    let s = sim.trajectory_sample(&command).unwrap();
    s.validate(6, None).unwrap();
    assert_eq!(s.command_time, 0.0);
    assert_eq!(s.measurement_time, sim.time());
    assert_eq!(s.source_measurement_time, 0.0);
    assert_eq!(s.applied_joint_position, home);
    assert_eq!(s.commanded_joint_position, command.position);
    assert_ne!(s.applied_joint_position, s.commanded_joint_position);
}
