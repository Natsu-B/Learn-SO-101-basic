//! Optional SDK adapter; no MuJoCo or Python dependency, no embedded GUI.
#[cfg(feature = "sdk")]
mod sdk {
    use robotics_core::{Error, Result, telemetry::TelemetrySink};
    use std::path::Path;
    pub struct RerunTelemetry {
        recording: rerun::RecordingStream,
        episode: i64,
        global_step: i64,
    }
    fn error(error: impl std::fmt::Display) -> Error {
        Error::Input(format!("Rerun telemetry: {error}"))
    }
    impl RerunTelemetry {
        pub fn spawn() -> Result<Self> {
            Ok(Self {
                episode: 0,
                global_step: 0,
                recording: rerun::RecordingStreamBuilder::new("robot-sim")
                    .spawn()
                    .map_err(error)?,
            })
        }
        pub fn save(path: impl AsRef<Path>) -> Result<Self> {
            Ok(Self {
                episode: 0,
                global_step: 0,
                recording: rerun::RecordingStreamBuilder::new("robot-sim")
                    .save(path.as_ref())
                    .map_err(error)?,
            })
        }
    }
    impl TelemetrySink for RerunTelemetry {
        fn begin_episode(&mut self, episode: u64) -> Result<()> {
            self.episode = i64::try_from(episode).map_err(error)?;
            Ok(())
        }
        fn begin_step(&mut self, time: f64, step: usize) -> Result<()> {
            self.recording.set_time(
                "simulation_time",
                std::time::Duration::try_from_secs_f64(time).map_err(error)?,
            );
            self.recording.set_time(
                "episode_step",
                rerun::TimeCell::from_sequence(i64::try_from(step).map_err(error)?),
            );
            self.recording
                .set_time("episode", rerun::TimeCell::from_sequence(self.episode));
            self.recording.set_time(
                "global_step",
                rerun::TimeCell::from_sequence(self.global_step),
            );
            self.global_step = self
                .global_step
                .checked_add(1)
                .ok_or_else(|| error("telemetry step overflow"))?;
            self.scalar("metrics/episode", self.episode as f64)?;
            self.scalar("simulation/time", time)
        }
        fn begin_frame(&mut self, time: f64) -> Result<()> {
            self.begin_step(time, 0)
        }
        fn scalar(&mut self, path: &str, value: f64) -> Result<()> {
            self.recording
                .log(path, &rerun::Scalars::new([value]))
                .map_err(error)
        }
        fn point3(&mut self, path: &str, position: [f64; 3]) -> Result<()> {
            self.recording
                .log(
                    path,
                    &rerun::Points3D::new([position.map(|v| v as f32)]).with_radii([0.015]),
                )
                .map_err(error)
        }
        fn flush(&mut self) -> Result<()> {
            self.recording.flush_blocking().map_err(error)
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn writes_headless_recording() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("test.rrd");
            let mut sink = RerunTelemetry::save(&path).unwrap();
            sink.begin_frame(0.5).unwrap();
            sink.scalar("robot/joint", 0.2).unwrap();
            sink.point3("world/target", [0.1, 0.2, 0.3]).unwrap();
            sink.flush().unwrap();
            drop(sink);
            assert!(std::fs::metadata(path).unwrap().len() > 100);
        }
    }
}
#[cfg(feature = "sdk")]
pub use sdk::RerunTelemetry;
