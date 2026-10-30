use so101_core::{
    Simulator,
    controller::{Controller, SineController},
};

fn main() -> so101_core::Result<()> {
    let mut sim = Simulator::new()?;
    let mut controller =
        SineController::new(sim.home_position(), &sim.action_bounds(), 0, 0.3, 0.25)?;
    for _ in 0..1000 {
        let command = controller.command(&sim.state(), sim.time());
        sim.step(&command)?;
    }
    println!("time: {}; state: {:?}", sim.time(), sim.state());
    Ok(())
}
