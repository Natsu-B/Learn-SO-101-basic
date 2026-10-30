use mujoco_runtime::{JointCommand, Simulator, presets};
use robotics_core::task::{ReachTask, Task};

#[test]
fn same_reach_task_for_three_robots_with_seeded_reset() {
    for name in ["so101", "panda", "ur5e"] {
        let mut sim = Simulator::from_spec(presets::robot(name).unwrap()).unwrap();
        let mut task = ReachTask::default();
        task.reset(&sim, 42).unwrap();
        let initial = sim.state();
        let target = task.target();
        let center = sim.end_effector("gripper").unwrap();
        let distance = target
            .iter()
            .zip(center)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((0.05..=0.25).contains(&distance));
        let witness = task.target_configuration();
        assert_eq!(sim.forward_kinematics(witness, "gripper").unwrap(), target);
        assert_eq!(sim.state(), initial);
        assert_eq!(sim.time(), 0.0);
        let command = JointCommand {
            position: initial.position.clone(),
        };
        let reward = task.reward(&sim).unwrap();
        assert!(reward.is_finite() && reward <= 0.0);
        for _ in 0..100 {
            sim.step(&command).unwrap();
        }
        sim.reset().unwrap();
        task.reset(&sim, 42).unwrap();
        assert_eq!(task.target(), target);
        assert_eq!(sim.state(), initial);
        assert_eq!(task.reward(&sim).unwrap(), reward);
        task.reset(&sim, 43).unwrap();
        assert_ne!(task.target(), target);
    }
}

#[test]
fn reward_is_negative_distance_and_success_uses_threshold() {
    for name in ["so101", "panda", "ur5e"] {
        let sim = Simulator::from_spec(presets::robot(name).unwrap()).unwrap();
        let mut task = ReachTask::default();
        let p = sim.end_effector("gripper").unwrap();
        task.set_target(p).unwrap();
        assert_eq!(task.reward(&sim).unwrap(), 0.0);
        assert!(task.terminated(&sim).unwrap());
        task.set_target([p[0] + 0.1, p[1], p[2]]).unwrap();
        assert!((task.reward(&sim).unwrap() + 0.1).abs() < 1e-12);
        assert!(!task.terminated(&sim).unwrap());
    }
}

#[test]
fn task_reset_does_not_reset_robot_and_sampling_failure_is_explicit() {
    use robotics_core::task::ReachConfig;
    let mut sim = Simulator::new().unwrap();
    sim.step(&JointCommand {
        position: sim.home_position(),
    })
    .unwrap();
    let state = sim.state();
    let time = sim.time();
    let mut task = ReachTask::default();
    task.reset(&sim, 7).unwrap();
    assert_eq!(state, sim.state());
    assert_eq!(time, sim.time());
    let mut impossible = ReachTask::new(ReachConfig {
        min_target_distance: 100.0,
        max_target_distance: 101.0,
        max_target_attempts: 3,
        ..Default::default()
    })
    .unwrap();
    assert!(
        impossible
            .reset(&sim, 7)
            .unwrap_err()
            .to_string()
            .contains("after 3 attempts")
    );
}

#[test]
fn bonus_success_and_time_limit_are_distinct() {
    use robotics_core::task::TaskConfig;
    let sim = Simulator::new().unwrap();
    let mut task = ReachTask::new(TaskConfig {
        success_bonus: 2.0,
        max_episode_steps: 2,
        ..Default::default()
    })
    .unwrap();
    task.reset(&sim, 42).unwrap();
    task.advance();
    assert!(!task.truncated(&sim).unwrap());
    task.advance();
    assert!(task.truncated(&sim).unwrap());
    assert!(!task.terminated(&sim).unwrap());
    task.set_target(sim.end_effector("gripper").unwrap())
        .unwrap();
    assert!(task.terminated(&sim).unwrap());
    assert!(!task.truncated(&sim).unwrap());
    assert_eq!(task.reward(&sim).unwrap(), 2.0);
}
#[test]
fn robot_io_does_not_advance_physics() {
    use robotics_core::{RobotAction, robot::RobotIo};
    let mut sim = Simulator::new().unwrap();
    let before = sim.observation().unwrap();
    sim.send_action(&RobotAction::JointPosition(vec![0.1; 6]))
        .unwrap();
    assert_eq!(sim.time(), 0.0);
    assert_eq!(sim.observation().unwrap(), before);
}

#[test]
fn task_uses_spec_default_effector() {
    let mut spec = mujoco_runtime::presets::robot("panda").unwrap();
    spec.end_effectors[0].name = "tool0".into();
    spec.default_end_effector = Some("tool0".into());
    let sim = Simulator::from_spec(spec).unwrap();
    let mut task = ReachTask::default();
    task.reset(&sim, 42).unwrap();
    assert!(task.reward(&sim).unwrap().is_finite());
}

#[test]
fn selected_joint_fk_and_dls_keep_omitted_joint_fixed() {
    let mut sim = Simulator::new().unwrap();
    let initial = sim.state();
    let mut task = ReachTask::default();
    task.set_fixed_joints(vec![(5, sim.home_position()[5])]);
    task.reset(&sim, 1_000_000).unwrap();
    assert_eq!(task.target_configuration()[5], sim.home_position()[5]);
    assert_eq!(initial, sim.state());
    assert_eq!(sim.time(), 0.0);
    for _ in 0..300 {
        let ranges = sim.action_bounds();
        let q: Vec<_> = sim
            .state()
            .position
            .iter()
            .zip(&ranges)
            .map(|(q, [lo, hi])| q.clamp(*lo, *hi))
            .collect();
        let dq = sim
            .dls_delta(&q, task.target(), &[0, 1, 2, 3, 4], 0.01, 0.05)
            .unwrap();
        assert_eq!(dq[5], 0.0);
        let mut command: Vec<_> = q
            .iter()
            .zip(dq)
            .zip(ranges)
            .map(|((q, dq), [lo, hi])| (q + dq).clamp(lo, hi))
            .collect();
        command[5] = sim.home_position()[5];
        sim.step_control(&JointCommand { position: command }, 1.0 / 30.0)
            .unwrap();
        if task.distance(&sim).unwrap() <= 0.005 {
            break;
        }
    }
    assert!(task.distance(&sim).unwrap() <= 0.005);
}
