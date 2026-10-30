use robotics_core::{Error, Result, trajectory::*};
use std::{
    fs::File,
    path::{Path, PathBuf},
};
fn input(e: impl std::fmt::Display) -> Error {
    Error::Input(e.to_string())
}
pub struct CsvRecorder {
    directory: PathBuf,
    writer: Option<csv::Writer<File>>,
    dimension: usize,
    previous: Option<f64>,
}
impl CsvRecorder {
    pub fn new(directory: impl AsRef<Path>) -> Self {
        Self {
            directory: directory.as_ref().to_owned(),
            writer: None,
            dimension: 0,
            previous: None,
        }
    }
}
pub fn header(names: &[String], version: u32) -> Vec<String> {
    let mut h: Vec<String> = if version == 1 {
        vec!["timestamp".into()]
    } else {
        [
            "measurement_time",
            "command_time",
            "applied_time",
            "source_measurement_time",
        ]
        .map(String::from)
        .to_vec()
    };
    let fields: &[&str] = if version == 1 {
        &["command", "position", "velocity"]
    } else {
        &["command", "applied", "position", "velocity"]
    };
    for field in fields {
        h.extend(names.iter().map(|n| format!("{field}/{n}")));
    }
    h.extend(["end_effector/x", "end_effector/y", "end_effector/z"].map(String::from));
    h
}
impl TrajectoryRecorder for CsvRecorder {
    fn begin(&mut self, metadata: &TrajectoryMetadata) -> Result<()> {
        metadata.validate()?;
        if metadata.schema_version != 2 {
            return Err(input("new recordings require schema V2"));
        }
        if self.writer.is_some() {
            return Err(input("recorder already begun"));
        }
        std::fs::create_dir(&self.directory)?; // Refuse to overwrite existing measurements.
        let metadata_file = File::create(self.directory.join("metadata.json"))?;
        serde_json::to_writer_pretty(metadata_file, metadata).map_err(input)?;
        let mut writer =
            csv::Writer::from_path(self.directory.join("samples.csv")).map_err(input)?;
        writer
            .write_record(header(&metadata.joint_names, metadata.schema_version))
            .map_err(input)?;
        self.dimension = metadata.joint_names.len();
        self.writer = Some(writer);
        self.previous = None;
        Ok(())
    }
    fn record(&mut self, sample: &TrajectorySample) -> Result<()> {
        sample.validate(self.dimension, self.previous)?;
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| input("call begin before record"))?;
        let mut row = [
            sample.measurement_time,
            sample.command_time,
            sample.applied_time,
            sample.source_measurement_time,
        ]
        .map(|t| t.to_string())
        .to_vec();
        row.extend(
            sample
                .commanded_joint_position
                .iter()
                .map(ToString::to_string),
        );
        row.extend(
            sample
                .applied_joint_position
                .iter()
                .map(ToString::to_string),
        );
        row.extend(
            sample
                .measured_joint_position
                .iter()
                .map(ToString::to_string),
        );
        row.extend((0..self.dimension).map(|i| {
            sample
                .measured_joint_velocity
                .as_ref()
                .map(|v| v[i].to_string())
                .unwrap_or_default()
        }));
        row.extend((0..3).map(|i| {
            sample
                .end_effector_position
                .map(|v| v[i].to_string())
                .unwrap_or_default()
        }));
        writer.write_record(row).map_err(input)?;
        self.previous = Some(sample.measurement_time);
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
        }
        Ok(())
    }
}
pub fn load(directory: impl AsRef<Path>) -> Result<(TrajectoryMetadata, Vec<TrajectorySample>)> {
    let m: TrajectoryMetadata =
        serde_json::from_reader(File::open(directory.as_ref().join("metadata.json"))?)
            .map_err(input)?;
    m.validate()?;
    let mut reader =
        csv::Reader::from_path(directory.as_ref().join("samples.csv")).map_err(input)?;
    if reader
        .headers()
        .map_err(input)?
        .iter()
        .ne(header(&m.joint_names, m.schema_version)
            .iter()
            .map(String::as_str))
    {
        return Err(input("trajectory CSV header mismatch"));
    }
    let n = m.joint_names.len();
    let mut samples: Vec<TrajectorySample> = vec![];
    for row in reader.records() {
        let row = row.map_err(input)?;
        let values = |start: usize, count: usize| -> Result<Vec<f64>> {
            (start..start + count)
                .map(|i| row[i].parse::<f64>().map_err(input))
                .collect()
        };
        let optional = |start: usize, count: usize| -> Result<Option<Vec<f64>>> {
            if (start..start + count).all(|i| row[i].is_empty()) {
                Ok(None)
            } else {
                values(start, count).map(Some)
            }
        };
        let base = if m.schema_version == 1 { 1 } else { 4 };
        let position = base + if m.schema_version == 1 { n } else { 2 * n };
        let time: f64 = row[0].parse().map_err(input)?;
        let s = TrajectorySample {
            measurement_time: time,
            command_time: if m.schema_version == 1 {
                time
            } else {
                row[1].parse().map_err(input)?
            },
            applied_time: if m.schema_version == 1 {
                time
            } else {
                row[2].parse().map_err(input)?
            },
            source_measurement_time: if m.schema_version == 1 {
                time
            } else {
                row[3].parse().map_err(input)?
            },
            commanded_joint_position: values(base, n)?,
            applied_joint_position: values(
                if m.schema_version == 1 {
                    base
                } else {
                    base + n
                },
                n,
            )?,
            measured_joint_position: values(position, n)?,
            measured_joint_velocity: optional(position + n, n)?,
            end_effector_position: optional(position + 2 * n, 3)?.map(|v| [v[0], v[1], v[2]]),
        };
        s.validate(n, samples.last().map(|s| s.measurement_time))?;
        samples.push(s);
    }
    Ok((m, samples))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_and_reject_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("trace");
        let m = TrajectoryMetadata {
            schema_version: 2,
            robot: "fixture".into(),
            backend: "real".into(),
            model_version: "calibration1".into(),
            joint_names: vec!["j".into()],
            control_period: 0.1,
            physics_timestep: None,
            timestamp: 1.0,
            seed: 42,
            randomization_parameters: serde_json::json!({}),
        };
        let s = TrajectorySample {
            measurement_time: 0.1,
            command_time: 0.0,
            applied_time: 0.0,
            source_measurement_time: 0.1,
            applied_joint_position: vec![0.2],
            commanded_joint_position: vec![0.2],
            measured_joint_position: vec![0.1],
            measured_joint_velocity: None,
            end_effector_position: None,
        };
        let mut r = CsvRecorder::new(&out);
        r.begin(&m).unwrap();
        r.record(&s).unwrap();
        assert!(r.record(&s).is_err());
        let mut bad = s.clone();
        bad.measurement_time = 0.2;
        bad.measured_joint_position.clear();
        assert!(r.record(&bad).is_err());
        r.finish().unwrap();
        let (actual, samples) = load(out).unwrap();
        assert_eq!(actual, m);
        assert_eq!(samples, vec![s]);
    }
}
