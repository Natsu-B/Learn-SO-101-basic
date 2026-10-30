use crate::Result;

pub trait TelemetrySink {
    fn begin_episode(&mut self, _episode: u64) -> Result<()> {
        Ok(())
    }
    fn begin_step(&mut self, time: f64, _episode_step: usize) -> Result<()> {
        self.begin_frame(time)
    }
    fn begin_frame(&mut self, time: f64) -> Result<()>;
    fn scalar(&mut self, path: &str, value: f64) -> Result<()>;
    fn point3(&mut self, path: &str, position: [f64; 3]) -> Result<()>;
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
#[derive(Default)]
pub struct NullTelemetry;
impl TelemetrySink for NullTelemetry {
    fn begin_frame(&mut self, _: f64) -> Result<()> {
        Ok(())
    }
    fn scalar(&mut self, _: &str, _: f64) -> Result<()> {
        Ok(())
    }
    fn point3(&mut self, _: &str, _: [f64; 3]) -> Result<()> {
        Ok(())
    }
}
