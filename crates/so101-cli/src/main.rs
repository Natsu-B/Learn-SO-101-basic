use so101_core::{
    Simulator,
    controller::{Controller, HomeController, SineController},
};

#[cfg(feature = "viewer")]
mod viewer;

fn main() {
    if let Err(error) = run() {
        eprintln!("so101-cli: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut steps = 1000usize;
    let mut sine = false;
    let mut json = false;
    let mut show_viewer = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => steps = args.next().ok_or("--steps requires a count")?.parse()?,
            "--sine" => sine = true,
            "--json" => json = true,
            "--viewer" => show_viewer = true,
            "--help" | "-h" => {
                println!(
                    "Usage: so101-cli [--steps N] [--sine] [--json] [--viewer]\nDefault: 1000 headless home steps.\nViewer requires --features viewer."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    #[cfg(not(feature = "viewer"))]
    if show_viewer {
        return Err(
            "viewer is disabled; use cargo run -p so101-cli --features viewer -- --viewer".into(),
        );
    }
    let mut sim = Simulator::new()?;
    let mut controller: Box<dyn Controller> = if sine {
        Box::new(SineController::new(
            sim.home_position(),
            &sim.action_bounds(),
            0,
            0.3,
            0.25,
        )?)
    } else {
        Box::new(HomeController::new(
            sim.home_position(),
            &sim.action_bounds(),
        )?)
    };
    if !json {
        println!(
            "actuators: {:?}; timestep: {} s",
            sim.actuator_names(),
            sim.timestep()
        );
    }
    if show_viewer {
        #[cfg(feature = "viewer")]
        viewer::run(&mut sim, controller.as_mut(), steps)?;
    } else {
        for _ in 0..steps {
            let command = controller.command(&sim.state(), sim.time());
            sim.step(&command)?;
        }
    }
    let state = sim.state();
    if json {
        println!(
            "{{\"time\":{},\"position\":{:?},\"velocity\":{:?}}}",
            sim.time(),
            state.position,
            state.velocity
        );
    } else {
        println!("time: {:.6} s; state: {:?}", sim.time(), state);
    }
    Ok(())
}
