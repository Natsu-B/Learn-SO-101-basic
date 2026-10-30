#[path = "common/reach.rs"]
mod reach;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    reach::run("ur5e")
}
