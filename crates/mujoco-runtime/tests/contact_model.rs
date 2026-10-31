//! Pairwise contact environment: friction DR changes only the jaw-pad and
//! table sliding friction; condim/solref/priority stay as in the scene.
use mujoco_runtime::pick_place::{
    ContactModel, ContactOverrides, ContactParameters, ObjectRandomization, PickPlaceConfig,
    PickPlaceSim,
};
use mujoco_runtime::skills::{
    CONTROL_PERIOD, Criteria, ScriptedGrasp, Skill, SkillStatus, run_episode, scripted_skills,
};

fn env(model: ContactModel, overrides: ContactOverrides) -> PickPlaceSim {
    PickPlaceSim::new(PickPlaceConfig {
        contact_model: model,
        contact_overrides: overrides,
        ..PickPlaceConfig::default()
    })
    .unwrap()
}
fn pairwise() -> PickPlaceSim {
    env(ContactModel::Pairwise, ContactOverrides::default())
}
fn fixed(mass: f64, jaw: Option<f64>, floor: Option<f64>) -> ObjectRandomization {
    ObjectRandomization {
        mass: Some([mass, mass]),
        friction: jaw.map(|f| [f, f]),
        floor_friction: floor.map(|f| [f, f]),
    }
}

/// Resolved contact parameters (jaw, table) seen while the IK Grasp closes.
fn resolved(pp: &mut PickPlaceSim, seed: u64) -> (ContactParameters, ContactParameters) {
    pp.reset(seed).unwrap();
    let table = pp
        .object_contact_parameters()
        .into_iter()
        .find(|c| c.partner == "table")
        .expect("cube rests on the table");
    let mut grasp = ScriptedGrasp::new(Criteria::default());
    grasp.begin(pp).unwrap();
    while grasp.status(pp) == SkillStatus::Running {
        let c = grasp.act(pp).unwrap();
        pp.step(&c, CONTROL_PERIOD).unwrap();
        if let Some(j) = pp
            .object_contact_parameters()
            .into_iter()
            .find(|c| c.partner == "moving_jaw" && c.dist < 1e-4)
        {
            return (j, table);
        }
    }
    panic!("no jaw contact during the grasp");
}

#[test]
fn pairwise_without_randomization_is_the_legacy_scene() {
    let mut legacy = env(ContactModel::Legacy, ContactOverrides::default());
    let mut pp = pairwise();
    assert_eq!(legacy.object_physics(), pp.object_physics());
    let criteria = Criteria::default();
    for seed in [0, 7] {
        let a = run_episode(
            &mut legacy,
            &mut scripted_skills(&criteria),
            seed,
            |_, _| {},
        )
        .unwrap();
        let b = run_episode(&mut pp, &mut scripted_skills(&criteria), seed, |_, _| {}).unwrap();
        assert!(a.success && b.success);
        assert_eq!(a.cube_final, b.cube_final, "identical physics, bit for bit");
    }
    let (jaw, table) = resolved(&mut pp, 3);
    assert_eq!(
        (jaw.dim, jaw.friction[0], jaw.solref),
        (6, 1.0, [0.01, 1.0])
    );
    assert_eq!((table.dim, table.friction[0]), (4, 1.2));
    assert!((table.solref[0] - 0.015).abs() < 1e-12);
}

#[test]
fn jaw_and_floor_friction_are_independent_and_nothing_else_changes() {
    let mut pp = pairwise();
    pp.set_object_randomization(fixed(0.02, Some(0.1), None))
        .unwrap();
    let (jaw, table) = resolved(&mut pp, 3);
    assert_eq!(
        (jaw.dim, jaw.friction[0], jaw.solref),
        (6, 0.1, [0.01, 1.0])
    );
    assert_eq!((table.dim, table.friction[0]), (4, 1.2));
    assert!((table.solref[0] - 0.015).abs() < 1e-12);
    let p = pp.object_physics();
    assert_eq!(
        (p.priority, p.jaw_friction, p.floor_friction),
        (0, 0.1, 1.2)
    );

    pp.set_object_randomization(fixed(0.02, None, Some(0.1)))
        .unwrap();
    let (jaw, table) = resolved(&mut pp, 3);
    assert_eq!((jaw.dim, jaw.friction[0]), (6, 1.0));
    assert_eq!((table.dim, table.friction[0]), (4, 0.1));
    assert!((table.solref[0] - 0.015).abs() < 1e-12);

    // Back to no DR: the scene values return exactly.
    pp.set_object_randomization(ObjectRandomization::default())
        .unwrap();
    let (jaw, table) = resolved(&mut pp, 3);
    assert_eq!((jaw.friction[0], table.friction[0]), (1.0, 1.2));
}

#[test]
fn same_seed_same_placement_and_mass_as_legacy() {
    let mut legacy = env(ContactModel::Legacy, ContactOverrides::default());
    let mut pp = pairwise();
    let r = ObjectRandomization {
        mass: Some([0.02, 0.1]),
        friction: Some([0.05, 1.5]),
        floor_friction: None,
    };
    legacy.set_object_randomization(r).unwrap();
    pp.set_object_randomization(r).unwrap();
    for seed in 0..10 {
        legacy.reset(seed).unwrap();
        pp.reset(seed).unwrap();
        let (a, b) = (legacy.object_physics(), pp.object_physics());
        assert_eq!((a.mass, a.friction), (b.mass, b.jaw_friction));
        assert_eq!(legacy.cube_pose(), pp.cube_pose());
        assert_eq!(legacy.target(), pp.target());
    }
}

#[test]
fn overrides_set_condim_and_legacy_rejects_pairwise_options() {
    let mut pp = env(
        ContactModel::Pairwise,
        ContactOverrides {
            jaw_condim: Some(4),
            ..ContactOverrides::default()
        },
    );
    let (jaw, _) = resolved(&mut pp, 3);
    assert_eq!(jaw.dim, 4);

    let mut legacy = env(ContactModel::Legacy, ContactOverrides::default());
    assert!(
        legacy
            .set_object_randomization(fixed(0.03, None, Some(0.5)))
            .is_err()
    );
    assert!(
        PickPlaceSim::new(PickPlaceConfig {
            contact_overrides: ContactOverrides {
                jaw_condim: Some(4),
                ..ContactOverrides::default()
            },
            ..PickPlaceConfig::default()
        })
        .is_err()
    );
    assert!(
        PickPlaceSim::new(PickPlaceConfig {
            contact_model: ContactModel::Pairwise,
            contact_overrides: ContactOverrides {
                jaw_condim: Some(5),
                ..ContactOverrides::default()
            },
            ..PickPlaceConfig::default()
        })
        .is_err()
    );
}

#[test]
fn contact_forces_balance_gravity_at_rest() {
    let mut pp = pairwise();
    pp.set_object_randomization(fixed(0.05, None, Some(0.3)))
        .unwrap();
    pp.reset(1).unwrap();
    let hold = mujoco_runtime::JointCommand {
        position: pp.sim.state().position,
    };
    for _ in 0..30 {
        pp.step(&hold, CONTROL_PERIOD).unwrap();
    }
    let details = pp.contact_details();
    assert!(!details.is_empty() && details.iter().all(|c| c.partner == "table"));
    let fz: f64 = details.iter().map(|c| c.force[2]).sum();
    assert!(
        (fz - 0.05 * 9.81).abs() < 1e-3 * 0.05 * 9.81 + 1e-6,
        "fz = {fz}"
    );
    for c in &details {
        assert!(c.normal[2] > 0.999, "table normal points up into the cube");
    }
}
