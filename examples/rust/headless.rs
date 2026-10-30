use so101_core::{JointCommand, Simulator};

fn main() -> so101_core::Result<()> {
    let mut sim = Simulator::new()?;
    println!("initial: {:?}", sim.state());
    for _ in 0..1000 {
        sim.step(&JointCommand {
            position: vec![0.0; 6],
        })?;
    }
    println!("time: {}; state: {:?}", sim.time(), sim.state());
    Ok(())
}
