//! Compliant-object grasp-force environment (docs/FORCE_CONTROL.md §3–4).
use mujoco_runtime::force_grasp::{
    CompliantObject, ForceGraspRanges, ForceGraspSim, ForcePi, PENETRATION_LIMIT,
};
use mujoco_runtime::pick_place::{PickPlaceConfig, PickPlaceSim, SCENE_TEMPLATE};
use mujoco_runtime::presets;
use mujoco_runtime::skills::{CONTROL_PERIOD, Criteria, run_episode, scripted_skills};

fn env() -> ForceGraspSim {
    ForceGraspSim::new(ForceGraspRanges::default()).unwrap()
}

/// Close slowly until both jaws touch the compliant faces.
fn pinch(e: &mut ForceGraspSim) -> f64 {
    let mut cmd = e.reading().unwrap().gripper_command;
    for _ in 0..400 {
        cmd -= 0.002;
        e.step(cmd).unwrap();
        let r = e.reading().unwrap();
        if r.jaw_force.iter().all(|f| *f > 0.05) {
            return r.gripper_q;
        }
    }
    panic!("jaws never pinched the object");
}

#[test]
fn from_scene_with_the_pick_place_template_is_the_same_environment() {
    let mut a = PickPlaceSim::new(PickPlaceConfig::default()).unwrap();
    let spec = presets::robot("so101").unwrap();
    let mut b = PickPlaceSim::from_scene(spec, PickPlaceConfig::default(), SCENE_TEMPLATE).unwrap();
    let criteria = Criteria::default();
    let ra = run_episode(&mut a, &mut scripted_skills(&criteria), 3, |_, _| {}).unwrap();
    let rb = run_episode(&mut b, &mut scripted_skills(&criteria), 3, |_, _| {}).unwrap();
    assert!(ra.success);
    assert_eq!(ra.cube_final, rb.cube_final, "bit-identical physics");
}

#[test]
fn sampled_objects_stay_in_range_and_validate() {
    let r = ForceGraspRanges::default();
    assert!((r.stiffness_at(0.) - r.stiffness[0]).abs() < 1e-9);
    assert!((r.stiffness_at(1.) - r.stiffness[1]).abs() < 1e-9);
    for seed in 0..200 {
        let o = r.sample(seed);
        o.validate().unwrap();
        assert!((r.width[0]..=r.width[1]).contains(&o.width));
        assert!((r.gap[0]..=r.gap[1]).contains(&o.gap));
        assert!((r.stiffness[1]..=r.stiffness[0]).contains(&o.stiffness));
        assert_eq!(o, r.sample(seed), "reproducible by seed");
    }
    let bad = CompliantObject {
        travel: 0.02,
        ..CompliantObject::nominal()
    };
    assert!(
        bad.validate().is_err(),
        "travel deeper than the core is rejected"
    );
}

#[test]
fn reset_is_reproducible_and_does_not_disturb_the_object() {
    let mut e = env();
    let mut first = None;
    for _ in 0..2 {
        e.reset(11, None).unwrap();
        assert!(e.positioning_displacement < 0.5e-3, "jaws clear the object");
        let r = e.reading().unwrap();
        assert_eq!(r.jaw_force, [0., 0.], "open jaws do not touch it");
        for _ in 0..20 {
            e.step(r.gripper_q - 0.3).unwrap();
        }
        let r = e.reading().unwrap();
        match first {
            None => first = Some(r),
            Some(f) => assert_eq!(f, r, "same seed, same trajectory"),
        }
    }
}

#[test]
fn jaw_force_follows_the_explicit_spring_law_independent_of_mass() {
    let mut e = env();
    let mut penetration = Vec::new();
    for mass in [0.02, 0.1] {
        let o = CompliantObject {
            stiffness: 1000.,
            mass,
            ..CompliantObject::nominal()
        };
        e.reset(5, Some(o)).unwrap();
        let q_c = pinch(&mut e);
        for _ in 0..45 {
            e.step(q_c - 0.04).unwrap();
        }
        let r = e.reading().unwrap();
        for side in 0..2 {
            assert!(!r.bottomed[side]);
            assert!(r.deformation[side] > 1e-3, "faces compress by mm");
            let rel = (r.jaw_force[side] - r.spring_force[side]).abs() / r.spring_force[side];
            assert!(rel < 0.05, "F = k d + c d' (side {side}: {r:?})");
        }
        assert_eq!(r.rigid_force, [0., 0.], "jaws only touch the faces");
        assert!(
            r.penetration < 0.2e-3,
            "numerical penetration << deformation"
        );
        penetration.push(r.penetration);
    }
    let ratio = penetration[0] / penetration[1];
    assert!(
        ratio < 3.,
        "penetration depends weakly on object mass: {penetration:?}"
    );
}

#[test]
fn crushing_is_reported_as_bottoming_out() {
    let mut e = env();
    let o = CompliantObject {
        stiffness: 200.,
        ..CompliantObject::nominal()
    };
    e.reset(5, Some(o)).unwrap();
    for _ in 0..60 {
        e.step(-0.17).unwrap();
    }
    let r = e.reading().unwrap();
    assert!(
        r.bottomed.iter().all(|b| *b),
        "stall torque crushes a soft object"
    );
    assert!(r.jaw_force[0] > 10. * o.max_spring_force());
}

#[test]
fn force_pi_tracks_the_goal_without_crushing() {
    let mut e = env();
    let ranges = *e.ranges();
    for seed in [6_000_002, 6_000_008, 6_000_015] {
        let o = ranges.sample(seed);
        let goal = 0.5 * o.max_spring_force().min(8.);
        e.reset(seed, Some(o)).unwrap();
        let mut pi = ForcePi::default();
        let mut tail = Vec::new();
        for k in 0..150 {
            let r = e.reading().unwrap();
            let d = pi.act(r.jaw_force, goal, CONTROL_PERIOD).clamp(-0.03, 0.03);
            e.step(r.gripper_q + d).unwrap();
            let r = e.reading().unwrap();
            assert!(r.penetration < PENETRATION_LIMIT);
            if k >= 120 {
                tail.push(r.jaw_force);
            }
        }
        let mean = tail.iter().map(|f| 0.5 * (f[0] + f[1])).sum::<f64>() / tail.len() as f64;
        assert!(
            (mean - goal).abs() < 0.1 * goal,
            "seed {seed}: {mean} vs {goal}"
        );
        assert!(!e.reading().unwrap().bottomed.iter().any(|b| *b));
    }
}
