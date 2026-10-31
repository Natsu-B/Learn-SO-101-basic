//! Reset-time object mass/friction randomization: model consistency, seeding,
//! independence from placement, and physically sane resting behaviour.
use mujoco_runtime::JointCommand;
use mujoco_runtime::pick_place::{ObjectRandomization, PickPlaceConfig, PickPlaceSim};
use mujoco_runtime::skills::{CONTROL_PERIOD, Criteria, run_episode, scripted_skills};

fn env() -> PickPlaceSim {
    PickPlaceSim::new(PickPlaceConfig::default()).unwrap()
}
const RANGE: ObjectRandomization = ObjectRandomization {
    mass: Some([0.02, 0.05]),
    friction: Some([0.8, 1.5]),
    floor_friction: None,
};

#[test]
fn disabled_keeps_scene_values_and_placement() {
    let mut pp = env();
    let nominal = pp.nominal_object_physics();
    assert_eq!(
        (nominal.mass, nominal.friction, nominal.priority),
        (0.03, 1.2, 0)
    );
    pp.reset(5).unwrap();
    assert_eq!(pp.object_physics(), nominal);
    let placement = (pp.cube_pose().0, pp.target());

    // Enabling DR uses an independent stream: same cube/goal for the same seed.
    pp.set_object_randomization(RANGE).unwrap();
    pp.reset(5).unwrap();
    assert_eq!((pp.cube_pose().0, pp.target()), placement);
    assert_ne!(pp.object_physics(), nominal);

    // Disabling restores the compiled values exactly.
    pp.set_object_randomization(ObjectRandomization::default())
        .unwrap();
    pp.reset(5).unwrap();
    assert_eq!(pp.object_physics(), nominal);
}

#[test]
fn sampling_is_seeded_bounded_and_inertia_consistent() {
    let mut pp = env();
    pp.set_object_randomization(RANGE).unwrap();
    let nominal = pp.nominal_object_physics();
    let mut seen = Vec::new();
    for seed in 0..50 {
        pp.reset(seed).unwrap();
        let p = pp.object_physics();
        assert!((0.02..=0.05).contains(&p.mass) && (0.8..=1.5).contains(&p.friction));
        // Same shape: inertia scales with mass (box: I = m a^2 / 6).
        for (i, n) in p.inertia.iter().zip(nominal.inertia) {
            assert!((i / n - p.mass / nominal.mass).abs() < 1e-12);
        }
        // Object friction must beat the gripper pads' priority-1 contact params.
        assert!(p.priority >= 2);
        assert_eq!(p, pp.sample_object_physics(seed));
        seen.push(p.mass);
    }
    pp.reset(3).unwrap();
    let again = pp.object_physics();
    pp.reset(4).unwrap();
    pp.reset(3).unwrap();
    assert_eq!(pp.object_physics(), again);
    seen.dedup();
    assert!(seen.len() > 40, "masses should vary across seeds");
}

#[test]
fn invalid_ranges_are_rejected() {
    let mut pp = env();
    for r in [
        ObjectRandomization {
            mass: Some([0.05, 0.02]),
            friction: None,
            floor_friction: None,
        },
        ObjectRandomization {
            mass: Some([0.0, 0.02]),
            friction: None,
            floor_friction: None,
        },
        ObjectRandomization {
            mass: None,
            friction: Some([f64::NAN, 1.0]),
            floor_friction: None,
        },
    ] {
        assert!(pp.set_object_randomization(r).is_err());
    }
}

#[test]
fn extreme_objects_rest_without_drift_or_instability() {
    let mut pp = env();
    for (m, f) in [(0.02, 0.8), (0.05, 1.5), (0.2, 0.05), (0.001, 3.0)] {
        pp.set_object_randomization(ObjectRandomization {
            mass: Some([m, m]),
            friction: Some([f, f]),
            floor_friction: None,
        })
        .unwrap();
        pp.reset(11).unwrap();
        let start = pp.cube_pose().0;
        let hold = JointCommand {
            position: pp.sim.state().position,
        };
        for _ in 0..60 {
            pp.step(&hold, CONTROL_PERIOD).unwrap();
        }
        let s = pp.state();
        let drift = (0..3)
            .map(|i| (s.cube_position[i] - start[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(drift < 3e-4, "m={m} f={f}: drift {drift}");
        assert!(s.cube_linear_velocity.iter().all(|v| v.abs() < 1e-2));
        assert!(s.contacts.table && !s.contacts.pinched());
    }
}

#[test]
fn scripted_pick_and_place_works_inside_training_range() {
    let mut pp = env();
    pp.set_object_randomization(RANGE).unwrap();
    let c = Criteria::default();
    for seed in [0, 1, 2] {
        let r = run_episode(&mut pp, &mut scripted_skills(&c), seed, |_, _| {}).unwrap();
        assert!(r.success, "seed {seed}: {:?}", r.skills);
    }
}

#[test]
fn released_cube_falls_and_settles_on_table() {
    use mujoco_runtime::pick_place::GRIPPER;
    use mujoco_runtime::skills::{GRIPPER_OPEN, ScriptedGrasp, Skill, SkillStatus};
    let mut pp = env();
    for (m, f) in [(0.02, 0.8), (0.05, 1.5), (0.1, 0.4)] {
        pp.set_object_randomization(ObjectRandomization {
            mass: Some([m, m]),
            friction: Some([f, f]),
            floor_friction: None,
        })
        .unwrap();
        pp.reset(2).unwrap();
        let mut grasp = ScriptedGrasp::new(Criteria::default());
        grasp.begin(&pp).unwrap();
        while grasp.status(&pp) == SkillStatus::Running {
            let c = grasp.act(&pp).unwrap();
            pp.step(&c, CONTROL_PERIOD).unwrap();
        }
        assert_eq!(grasp.status(&pp), SkillStatus::Success, "m={m} f={f}");
        let lifted = pp.cube_pose().0[2];
        let mut open = pp.sim.applied_command().clone();
        open.position[GRIPPER] = GRIPPER_OPEN;
        for _ in 0..60 {
            pp.step(&open, CONTROL_PERIOD).unwrap();
        }
        let s = pp.state();
        let half = pp.config().cube_half;
        assert!(lifted > half + 0.03);
        // Back on the table, at rest, not penetrating or bouncing away.
        assert!(s.contacts.table && !s.contacts.pinched(), "m={m} f={f}");
        assert!(
            (s.cube_position[2] - half).abs() < 3e-3,
            "z={}",
            s.cube_position[2]
        );
        assert!(s.cube_linear_velocity.iter().all(|v| v.abs() < 2e-2));
    }
}

#[test]
fn sphere_scene_keeps_mass_and_rests_in_place() {
    use mujoco_runtime::pick_place::ObjectShape;
    let mut pp = PickPlaceSim::new(PickPlaceConfig {
        shape: ObjectShape::Sphere,
        ..Default::default()
    })
    .unwrap();
    let p = pp.nominal_object_physics();
    assert_eq!(p.mass, 0.03);
    // Solid sphere: I = 2/5 m r^2 about every axis.
    for i in p.inertia {
        assert!((i - 0.4 * 0.03 * 0.0125f64.powi(2)).abs() < 1e-12);
    }
    pp.reset(3).unwrap();
    let start = pp.cube_pose().0;
    let hold = JointCommand {
        position: pp.sim.state().position,
    };
    for _ in 0..60 {
        pp.step(&hold, CONTROL_PERIOD).unwrap();
    }
    let s = pp.state();
    let drift = (0..3)
        .map(|i| (s.cube_position[i] - start[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(drift < 3e-4, "sphere drifted {drift} m");
    assert!(s.contacts.table && !s.contacts.pinched());
}
