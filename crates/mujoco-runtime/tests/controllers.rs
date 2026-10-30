use mujoco_runtime::{Simulator, presets};
use robotics_core::controller::{Controller, HomeController, SineController};
#[test]
fn three_robot_homes_and_panda_sine_are_legal() {
    for robot in ["so101", "panda", "ur5e"] {
        let mut sim = Simulator::from_spec(presets::robot(robot).unwrap()).unwrap();
        let home = sim.home_position();
        let mut controller = HomeController::new(home.clone(), &sim.action_bounds()).unwrap();
        sim.step(&controller.command(&sim.state(), 0.0)).unwrap();
        if robot == "panda" {
            assert!(home[3] < 0.0);
        }
        let mut sine =
            SineController::new(home.clone(), &sim.action_bounds(), 2, 0.1, 0.25).unwrap();
        let command = sine.command(&sim.state(), 1.0);
        assert!((command.position[2] - home[2] - 0.1).abs() < 1e-12);
        for (i, q) in home.iter().enumerate() {
            if i != 2 {
                assert_eq!(command.position[i], *q);
            }
        }
        sim.step(&command).unwrap();
    }
}
#[test]
fn explicit_home_has_priority_and_invalid_home_is_rejected() {
    let mut spec = presets::robot("panda").unwrap();
    let mut home = Simulator::from_spec(spec.clone()).unwrap().home_position();
    home[0] = 0.2;
    spec.home_position = Some(home.clone());
    assert_eq!(
        Simulator::from_spec(spec.clone()).unwrap().home_position(),
        home
    );
    spec.home_position = Some(vec![0.0; 7]);
    assert!(Simulator::from_spec(spec).is_err());
}
