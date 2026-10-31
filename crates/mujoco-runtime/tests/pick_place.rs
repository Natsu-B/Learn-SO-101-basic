//! Physical pick-and-place checks. These verify behaviour in simulation
//! (contact, lifting, release), not only that the code compiles.
use mujoco_runtime::JointCommand;
use mujoco_runtime::pick_place::{GRIPPER, IkTarget, PickPlaceConfig, PickPlaceSim};
use mujoco_runtime::skills::{
    CONTROL_PERIOD, Criteria, GRIPPER_OPEN, ScriptedGrasp, Skill, SkillStatus, predicates,
    run_episode, scripted_skills,
};

fn env() -> PickPlaceSim {
    PickPlaceSim::new(PickPlaceConfig::default()).unwrap()
}

#[test]
fn cube_rests_on_table_and_only_touches_it() {
    let mut pp = env();
    pp.reset(3).unwrap();
    let start = pp.cube_pose().0;
    let home = JointCommand {
        position: pp.sim.state().position,
    };
    for _ in 0..60 {
        pp.step(&home, CONTROL_PERIOD).unwrap();
    }
    let s = pp.state();
    let drift = (0..3)
        .map(|i| (s.cube_position[i] - start[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(drift < 2e-4, "cube drifted {drift} m at rest");
    assert!(s.contacts.table && !s.contacts.pinched() && !s.contacts.other_robot);
    assert_eq!(pp.sim.actuator_names().len(), 6);
}

#[test]
fn reset_is_seeded_and_samples_separated_reachable_positions() {
    let mut pp = env();
    let c = PickPlaceConfig::default();
    for seed in 0..20 {
        pp.reset(seed).unwrap();
        let (cube, _) = pp.cube_pose();
        let t = pp.target();
        for p in [cube, t] {
            let r = p[0].hypot(p[1]);
            assert!(r >= c.radius_range[0] - 1e-9 && r <= c.radius_range[1] + 1e-9);
        }
        assert!((cube[0] - t[0]).hypot(cube[1] - t[1]) >= c.min_separation);
        assert_eq!(pp.sim.time(), 0.0);
    }
    pp.reset(7).unwrap();
    let a = (pp.cube_pose().0, pp.target());
    pp.reset(8).unwrap();
    pp.reset(7).unwrap();
    assert_eq!(a, (pp.cube_pose().0, pp.target()));
}

#[test]
fn top_down_ik_reaches_table_positions_with_vertical_fingers() {
    let pp = env();
    let q0 = pp.sim.state().position;
    for target in [[0.15, 0.0, 0.0125], [0.25, 0.0, 0.0125], [0.2, 0.15, 0.03]] {
        let (q, err) = pp
            .solve_ik(&q0, &IkTarget::top_down(target, None), 300)
            .unwrap();
        let (p, axis) = pp.gripper_fk(&q).unwrap();
        assert!(err < 1e-3, "IK error {err} for {target:?}");
        assert!(axis[2] < -0.995, "fingers not vertical: {axis:?}");
        assert!((0..3).all(|i| (p[i] - target[i]).abs() < 1e-3));
        assert_eq!(q[GRIPPER], q0[GRIPPER]);
    }
    // FK is side-effect free.
    assert_eq!(pp.sim.time(), 0.0);
    assert_eq!(pp.sim.state().position, q0);
}

fn run_grasp(pp: &mut PickPlaceSim, seed: u64, force_open: bool) -> SkillStatus {
    pp.reset(seed).unwrap();
    let mut grasp = ScriptedGrasp::new(Criteria::default());
    grasp.begin(pp).unwrap();
    loop {
        let status = grasp.status(pp);
        if status != SkillStatus::Running {
            return status;
        }
        let mut command = grasp.act(pp).unwrap();
        if force_open {
            command.position[GRIPPER] = GRIPPER_OPEN;
        }
        pp.step(&command, CONTROL_PERIOD).unwrap();
    }
}

#[test]
fn grasp_lifts_cube_through_jaw_contact() {
    let mut pp = env();
    let criteria = Criteria::default();
    let status = run_grasp(&mut pp, 11, false);
    assert_eq!(status, SkillStatus::Success);
    let s = pp.state();
    assert!(
        s.contacts.fixed_jaw && s.contacts.moving_jaw,
        "{:?}",
        s.contacts
    );
    assert!(!s.contacts.table);
    assert!(s.cube_position[2] > 0.0125 + criteria.lift_height);
    // The cube is held between the jaws, near the calibrated pinch centre.
    let offset = (0..3)
        .map(|i| (s.cube_position[i] - s.gripper_position[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(offset < 0.01, "cube {offset} m from pinch centre");
    assert!(predicates::lifted(&pp, &criteria));
}

#[test]
fn control_without_closing_does_not_lift_cube() {
    // Same motion with the gripper held open: no kinematic attachment exists,
    // so the cube must stay on the table.
    let mut pp = env();
    let status = run_grasp(&mut pp, 11, true);
    assert!(matches!(status, SkillStatus::Failure(_)), "{status:?}");
    let s = pp.state();
    assert!(s.cube_position[2] < 0.0125 + 0.005);
    assert!(!s.contacts.pinched());
}

#[test]
fn scripted_pick_and_place_succeeds_across_seeds_by_contact_only() {
    let mut pp = env();
    let criteria = Criteria::default();
    for seed in [0_u64, 25, 52, 75, 180] {
        let mut skills = scripted_skills(&criteria);
        let mut last = None;
        let mut max_jump: f64 = 0.;
        let report = run_episode(&mut pp, &mut skills, seed, |e, _| {
            let p = e.cube_pose().0;
            if let Some(q) = last {
                let q: [f64; 3] = q;
                let d = (0..3).map(|i| (p[i] - q[i]).powi(2)).sum::<f64>().sqrt();
                max_jump = max_jump.max(d);
            }
            last = Some(p);
        })
        .unwrap();
        assert!(report.success, "seed {seed}: {report:?}");
        assert_eq!(report.skills.len(), 3);
        assert!(report.skills.iter().all(|s| s.success && !s.learned));
        assert!(report.final_xy_error <= criteria.place_tolerance);
        // Physical motion: no teleport after reset. Carrying moves the cube up
        // to ~0.35 m/s (~12 mm per 33 ms step); 30 mm would be ~0.9 m/s.
        assert!(max_jump < 0.03, "seed {seed}: jump {max_jump}");
        assert!(predicates::placed(&pp, &criteria));
    }
}

#[test]
fn stabilize_after_grasp_settles_cube_at_carry_height() {
    use mujoco_runtime::skills::ScriptedStabilize;
    let mut pp = env();
    for best_effort in [false, true] {
        assert_eq!(run_grasp(&mut pp, 0, false), SkillStatus::Success);
        let mut s = ScriptedStabilize::new(Criteria::default());
        s.best_effort = best_effort;
        s.begin(&pp).unwrap();
        while s.status(&pp) == SkillStatus::Running {
            let c = s.act(&pp).unwrap();
            pp.step(&c, CONTROL_PERIOD).unwrap();
        }
        assert_eq!(s.status(&pp), SkillStatus::Success);
        let st = pp.state();
        let half = pp.config().cube_half;
        assert!(st.contacts.pinched() && (st.cube_position[2] - half - 0.08).abs() < 0.01);
        assert!(
            st.cube_linear_velocity
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                < 0.02
        );
    }
}
