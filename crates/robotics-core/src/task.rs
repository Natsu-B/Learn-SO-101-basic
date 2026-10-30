use crate::{Error, Result};

/// Geometry services for simulated tasks, separate from hardware Robot I/O.
/// FK must not modify the observed robot state or advance time.
pub trait ReachGeometry {
    fn default_end_effector(&self) -> Result<&str>;
    fn end_effector(&self, name: &str) -> Result<[f64; 3]>;
    fn action_bounds(&self) -> Vec<[f64; 2]>;
    fn forward_kinematics(&self, position: &[f64], effector: &str) -> Result<[f64; 3]>;
}
pub trait Task {
    /// Sample task state only. Environment owners explicitly reset simulation.
    fn reset(&mut self, geometry: &dyn ReachGeometry, seed: u64) -> Result<()>;
    fn reward(&self, geometry: &dyn ReachGeometry) -> Result<f64>;
    fn terminated(&self, geometry: &dyn ReachGeometry) -> Result<bool>;
}
#[derive(Debug, Clone)]
pub struct ReachConfig {
    pub success_threshold: f64,
    pub success_bonus: f64,
    pub max_episode_steps: usize,
    pub min_target_distance: f64,
    pub max_target_distance: f64,
    /// Fraction of each finite safe range excluded at both ends.
    pub joint_margin: f64,
    pub max_target_attempts: usize,
}
impl Default for ReachConfig {
    fn default() -> Self {
        Self {
            success_threshold: 0.02,
            success_bonus: 0.0,
            max_episode_steps: 1000,
            min_target_distance: 0.05,
            max_target_distance: 0.25,
            joint_margin: 0.05,
            max_target_attempts: 4096,
        }
    }
}
impl ReachConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.success_threshold.is_finite()
            || self.success_threshold <= 0.0
            || !self.success_bonus.is_finite()
            || self.success_bonus < 0.0
            || self.max_episode_steps == 0
            || !self.min_target_distance.is_finite()
            || !self.max_target_distance.is_finite()
            || self.min_target_distance < 0.0
            || self.max_target_distance <= self.min_target_distance
            || !self.joint_margin.is_finite()
            || !(0.0..0.5).contains(&self.joint_margin)
            || self.max_target_attempts == 0
        {
            return Err(Error::Input(
                "invalid Reach target sampling configuration".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct ReachTask {
    target: [f64; 3],
    target_configuration: Vec<f64>,
    config: ReachConfig,
    episode_steps: usize,
    fixed_joints: Vec<(usize, f64)>,
}
impl Default for ReachTask {
    fn default() -> Self {
        Self::new(ReachConfig::default()).expect("valid default config")
    }
}
impl ReachTask {
    pub fn new(config: ReachConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            target: [0.0; 3],
            target_configuration: vec![],
            config,
            episode_steps: 0,
            fixed_joints: vec![],
        })
    }
    pub fn fixed_joints(&self) -> &[(usize, f64)] {
        &self.fixed_joints
    }
    pub fn set_fixed_joints(&mut self, joints: Vec<(usize, f64)>) {
        self.fixed_joints = joints;
    }
    pub fn episode_steps(&self) -> usize {
        self.episode_steps
    }
    pub fn advance(&mut self) {
        self.episode_steps = self.episode_steps.saturating_add(1);
    }
    pub fn truncated(&self, geometry: &dyn ReachGeometry) -> Result<bool> {
        Ok(self.episode_steps >= self.config.max_episode_steps && !self.terminated(geometry)?)
    }
    pub fn target(&self) -> [f64; 3] {
        self.target
    }
    /// FK witness, useful for auditing target reachability. Not a policy observation.
    pub fn target_configuration(&self) -> &[f64] {
        &self.target_configuration
    }
    pub fn set_target(&mut self, target: [f64; 3]) -> Result<()> {
        if !target.iter().all(|v| v.is_finite()) {
            return Err(Error::Input("non-finite Reach target".into()));
        }
        self.target = target;
        self.target_configuration.clear();
        Ok(())
    }
    pub fn distance(&self, geometry: &dyn ReachGeometry) -> Result<f64> {
        Ok(distance(
            geometry.end_effector(geometry.default_end_effector()?)?,
            self.target,
        ))
    }
}
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1_u64 << 53) as f64)
}
impl Task for ReachTask {
    fn reset(&mut self, geometry: &dyn ReachGeometry, seed: u64) -> Result<()> {
        let center = geometry.end_effector(geometry.default_end_effector()?)?;
        let bounds = geometry.action_bounds();
        if bounds.is_empty()
            || bounds
                .iter()
                .any(|[a, b]| !a.is_finite() || !b.is_finite() || a >= b || !(b - a).is_finite())
        {
            return Err(Error::Input(
                "Reach sampling requires finite, nonempty safe joint ranges".into(),
            ));
        }
        let mut rng = seed;
        for _ in 0..self.config.max_target_attempts {
            let mut q: Vec<_> = bounds
                .iter()
                .map(|[lo, hi]| {
                    lo + (hi - lo)
                        * (self.config.joint_margin
                            + (1.0 - 2.0 * self.config.joint_margin) * uniform(&mut rng))
                })
                .collect();
            for &(index, value) in &self.fixed_joints {
                if index >= q.len()
                    || !value.is_finite()
                    || value < bounds[index][0]
                    || value > bounds[index][1]
                {
                    return Err(Error::Input("invalid task joint selection".into()));
                }
                q[index] = value;
            }
            let target = geometry.forward_kinematics(&q, geometry.default_end_effector()?)?;
            let d = distance(center, target);
            if d >= self.config.min_target_distance && d <= self.config.max_target_distance {
                self.episode_steps = 0;
                self.target = target;
                self.target_configuration = q;
                return Ok(());
            }
        }
        Err(Error::Input(format!(
            "no FK Reach target within [{}, {}] m after {} attempts",
            self.config.min_target_distance,
            self.config.max_target_distance,
            self.config.max_target_attempts
        )))
    }
    fn reward(&self, geometry: &dyn ReachGeometry) -> Result<f64> {
        let d = self.distance(geometry)?;
        Ok(-d
            + if d <= self.config.success_threshold {
                self.config.success_bonus
            } else {
                0.0
            })
    }
    fn terminated(&self, geometry: &dyn ReachGeometry) -> Result<bool> {
        Ok(self.distance(geometry)? <= self.config.success_threshold)
    }
}

/// Shared task and episode configuration (SI distances, Gym/control step horizon).
pub type TaskConfig = ReachConfig;

/// Coherent task snapshot captured in one backend transaction.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub joint_state: crate::JointState,
    pub end_effector_position: [f64; 3],
    pub target_position: [f64; 3],
    pub reward: f64,
    pub distance: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub simulation_time: f64,
    pub observation_time: f64,
    pub episode_step: usize,
}
