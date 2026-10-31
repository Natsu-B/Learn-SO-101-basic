//! Pick-and-place skills (Grasp -> Transport -> Place) behind one interface.
//! Success predicates live here so scripted (IK) skills, learned policies and
//! evaluation all use identical definitions.
use crate::pick_place::{GRIPPER, IkTarget, PickPlaceSim};
use robotics_core::{JointCommand, Result};
use serde::Serialize;

pub const CONTROL_PERIOD: f64 = 1.0 / 30.0;
/// Gripper position targets (rad). Closing beyond contact makes the servo squeeze
/// at its force limit; the jaw stops on the cube.
pub const GRIPPER_OPEN: f64 = 0.6;
pub const GRIPPER_CLOSED: f64 = -0.17;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum SkillStatus {
    Running,
    Success,
    Failure(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct Criteria {
    /// Grasp: cube centre lifted this far above its resting height, while pinched.
    pub lift_height: f64,
    /// Grasp: lifted and pinched continuously for this long (s).
    pub hold_seconds: f64,
    /// Transport: XY distance between cube and goal while still pinched and lifted.
    pub transport_tolerance: f64,
    /// Place: final XY error after release.
    pub place_tolerance: f64,
    /// Place: cube linear speed below this counts as resting (m/s).
    pub rest_speed: f64,
}
impl Default for Criteria {
    fn default() -> Self {
        Self {
            lift_height: 0.03,
            hold_seconds: 0.3,
            transport_tolerance: 0.015,
            place_tolerance: 0.02,
            rest_speed: 0.01,
        }
    }
}

fn xy_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Instantaneous predicates on the true simulator state.
pub mod predicates {
    use super::*;
    pub fn lifted(env: &PickPlaceSim, c: &Criteria) -> bool {
        let s = env.state();
        s.contacts.pinched()
            && !s.contacts.table
            && s.cube_position[2] >= env.config().cube_half + c.lift_height
    }
    pub fn transported(env: &PickPlaceSim, c: &Criteria) -> bool {
        let s = env.state();
        s.contacts.pinched()
            && s.cube_position[2] >= env.config().cube_half + 0.5 * c.lift_height
            && xy_distance(s.cube_position, s.target_position) <= c.transport_tolerance
    }
    pub fn placed(env: &PickPlaceSim, c: &Criteria) -> bool {
        let s = env.state();
        !s.contacts.fixed_jaw
            && !s.contacts.moving_jaw
            && s.contacts.table
            && (s.cube_position[2] - env.config().cube_half).abs() < 0.003
            && norm(s.cube_linear_velocity) < c.rest_speed
            && xy_distance(s.cube_position, s.target_position) <= c.place_tolerance
    }
    /// Cube left the gripper and is back near the table.
    pub fn dropped(env: &PickPlaceSim) -> bool {
        let s = env.state();
        !s.contacts.pinched() && s.cube_position[2] < env.config().cube_half + 0.01
    }
}

pub trait Skill {
    fn name(&self) -> &'static str;
    /// Whether this skill's actions come from a learned model.
    fn learned(&self) -> bool {
        false
    }
    /// Name of the internal phase, for skills that have one (diagnostics and
    /// for starting learned skills from a scripted phase boundary).
    fn phase(&self) -> Option<&'static str> {
        None
    }
    fn begin(&mut self, env: &PickPlaceSim) -> Result<()>;
    fn act(&mut self, env: &PickPlaceSim) -> Result<JointCommand>;
    fn status(&self, env: &PickPlaceSim) -> SkillStatus;
}

/// Shared Cartesian motion helper: a speed-limited set point, warm-started IK,
/// and a per-step joint change limit so the high-gain servos are not excited.
#[derive(Debug, Clone)]
struct Motion {
    setpoint: [f64; 3],
    ik_q: Vec<f64>,
    command: Vec<f64>,
    closing: Option<[f64; 3]>,
    speed: f64,
    max_joint_step: f64,
}
impl Motion {
    fn new(env: &PickPlaceSim, closing: Option<[f64; 3]>) -> Self {
        let s = env.state();
        let command = env.sim.applied_command().position.clone();
        Self {
            setpoint: s.gripper_position,
            ik_q: command.clone(),
            command,
            closing,
            speed: 0.15,
            max_joint_step: 0.05,
        }
    }
    fn track(&mut self, env: &PickPlaceSim, goal: [f64; 3], gripper: f64) -> Result<JointCommand> {
        let d = [
            goal[0] - self.setpoint[0],
            goal[1] - self.setpoint[1],
            goal[2] - self.setpoint[2],
        ];
        let n = norm(d);
        let step = self.speed * CONTROL_PERIOD;
        if n <= step {
            self.setpoint = goal;
        } else {
            for (s, d) in self.setpoint.iter_mut().zip(d) {
                *s += d * step / n;
            }
        }
        let (q, _) = env.solve_ik(
            &self.ik_q,
            &IkTarget::top_down(self.setpoint, self.closing),
            60,
        )?;
        self.ik_q = q.clone();
        let bounds = env.sim.action_bounds();
        for (i, (c, t)) in self.command.iter_mut().zip(&q).enumerate() {
            if i == GRIPPER {
                continue;
            }
            *c = (*c + (t - *c).clamp(-self.max_joint_step, self.max_joint_step))
                .clamp(bounds[i][0], bounds[i][1]);
        }
        self.command[GRIPPER] = gripper.clamp(bounds[GRIPPER][0], bounds[GRIPPER][1]);
        Ok(JointCommand {
            position: self.command.clone(),
        })
    }
    /// Set point arrived and the measured pinch centre is close to it.
    fn arrived(&self, env: &PickPlaceSim, goal: [f64; 3], tolerance: f64) -> bool {
        self.setpoint == goal && norm(sub(env.state().gripper_position, goal)) <= tolerance
    }
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Cube face normal (horizontal) closest to the arm's natural closing direction.
fn grasp_closing(env: &PickPlaceSim, above: [f64; 3]) -> Result<[f64; 3]> {
    let q0 = env.sim.applied_command().position.clone();
    let (q, _) = env.solve_ik(&q0, &IkTarget::top_down(above, None), 200)?;
    // Closing direction of the unconstrained top-down solution.
    let natural = env.fk_closing(&q)?;
    let (_, quat) = env.cube_pose();
    let yaw = 2.0 * quat[3].atan2(quat[0]);
    let candidates = [[yaw.cos(), yaw.sin(), 0.], [-yaw.sin(), yaw.cos(), 0.]];
    Ok(*candidates
        .iter()
        .max_by(|a, b| {
            let da = (a[0] * natural[0] + a[1] * natural[1]).abs();
            let db = (b[0] * natural[0] + b[1] * natural[1]).abs();
            da.total_cmp(&db)
        })
        .unwrap())
}

// ---------------------------------------------------------------- Grasp
#[derive(Debug, Clone, Copy, PartialEq)]
enum GraspPhase {
    Approach,
    Descend,
    Close,
    Lift,
    Hold,
}
pub struct ScriptedGrasp {
    criteria: Criteria,
    motion: Option<Motion>,
    phase: GraspPhase,
    phase_steps: usize,
    steps: usize,
    held: f64,
    cube_start: [f64; 3],
    /// Lateral shift away from the fixed jaw while descending (world, metres).
    clearance: [f64; 3],
    pub approach_height: f64,
    /// Fixed-jaw clearance to the cube face during descent (metres).
    pub jaw_clearance: f64,
    pub carry_height: f64,
    pub max_steps: usize,
}
impl ScriptedGrasp {
    pub fn new(criteria: Criteria) -> Self {
        Self {
            criteria,
            motion: None,
            phase: GraspPhase::Approach,
            phase_steps: 0,
            steps: 0,
            held: 0.,
            cube_start: [0.; 3],
            clearance: [0.; 3],
            approach_height: 0.06,
            jaw_clearance: 0.004,
            carry_height: 0.08,
            max_steps: 300,
        }
    }
    fn next(&mut self, phase: GraspPhase) {
        self.phase = phase;
        self.phase_steps = 0;
    }
}
impl Skill for ScriptedGrasp {
    fn name(&self) -> &'static str {
        "grasp"
    }
    fn phase(&self) -> Option<&'static str> {
        Some(match self.phase {
            GraspPhase::Approach => "approach",
            GraspPhase::Descend => "descend",
            GraspPhase::Close => "close",
            GraspPhase::Lift => "lift",
            GraspPhase::Hold => "hold",
        })
    }
    fn begin(&mut self, env: &PickPlaceSim) -> Result<()> {
        self.cube_start = env.cube_pose().0;
        let above = add(self.cube_start, [0., 0., self.approach_height]);
        let closing = grasp_closing(env, above)?;
        self.motion = Some(Motion::new(env, Some(closing)));
        self.phase = GraspPhase::Approach;
        self.phase_steps = 0;
        self.steps = 0;
        self.held = 0.;
        self.clearance = [0.; 3];
        Ok(())
    }
    fn act(&mut self, env: &PickPlaceSim) -> Result<JointCommand> {
        self.steps += 1;
        self.phase_steps += 1;
        // The fixed jaw is static and sits ~1 mm from the cube face at the pinch
        // centre; descend with the fixed jaw shifted away from the cube. The
        // moving jaw then pushes the cube against the fixed jaw while closing.
        let cube = add(self.cube_start, self.clearance);
        let above = add(cube, [0., 0., self.approach_height]);
        let lifted = [cube[0], cube[1], env.config().cube_half + self.carry_height];
        let motion = self.motion.as_mut().expect("begin() first");
        let command = match self.phase {
            GraspPhase::Approach => {
                let c = motion.track(env, above, GRIPPER_OPEN)?;
                if motion.arrived(env, above, 0.004) || self.phase_steps > 120 {
                    // Site +z points from the fixed jaw towards the moving jaw;
                    // moving along -z takes the fixed jaw away from the cube face.
                    let z = env.closing_direction();
                    let h = z[0].hypot(z[1]).max(1e-9);
                    self.clearance = [
                        -self.jaw_clearance * z[0] / h,
                        -self.jaw_clearance * z[1] / h,
                        0.,
                    ];
                    self.next(GraspPhase::Descend);
                }
                c
            }
            GraspPhase::Descend => {
                motion.speed = 0.08;
                let c = motion.track(env, cube, GRIPPER_OPEN)?;
                if motion.arrived(env, cube, 0.003) || self.phase_steps > 60 {
                    self.next(GraspPhase::Close);
                }
                c
            }
            GraspPhase::Close => {
                let c = motion.track(env, cube, GRIPPER_CLOSED)?;
                if self.phase_steps >= 15 {
                    self.next(GraspPhase::Lift);
                }
                c
            }
            GraspPhase::Lift => {
                motion.speed = 0.10;
                let c = motion.track(env, lifted, GRIPPER_CLOSED)?;
                if motion.arrived(env, lifted, 0.01) || self.phase_steps > 60 {
                    self.next(GraspPhase::Hold);
                }
                c
            }
            GraspPhase::Hold => motion.track(env, lifted, GRIPPER_CLOSED)?,
        };
        if self.phase == GraspPhase::Hold && predicates::lifted(env, &self.criteria) {
            self.held += CONTROL_PERIOD;
        } else {
            self.held = 0.;
        }
        Ok(command)
    }
    fn status(&self, env: &PickPlaceSim) -> SkillStatus {
        if self.held >= self.criteria.hold_seconds && predicates::lifted(env, &self.criteria) {
            SkillStatus::Success
        } else if self.steps >= self.max_steps {
            SkillStatus::Failure(format!("timeout in {:?}", self.phase))
        } else if self.phase == GraspPhase::Hold && self.phase_steps > 30 {
            SkillStatus::Failure("cube not held after lift".into())
        } else {
            SkillStatus::Running
        }
    }
}

// ------------------------------------------------------------ Transport
pub struct ScriptedTransport {
    criteria: Criteria,
    motion: Option<Motion>,
    /// Cube position relative to the pinch centre, measured at begin().
    hold_offset: [f64; 3],
    steps: usize,
    pub carry_height: f64,
    pub max_steps: usize,
}
impl ScriptedTransport {
    pub fn new(criteria: Criteria) -> Self {
        Self {
            criteria,
            motion: None,
            hold_offset: [0.; 3],
            steps: 0,
            carry_height: 0.08,
            max_steps: 200,
        }
    }
}
impl Skill for ScriptedTransport {
    fn name(&self) -> &'static str {
        "transport"
    }
    fn begin(&mut self, env: &PickPlaceSim) -> Result<()> {
        let s = env.state();
        self.hold_offset = sub(s.cube_position, s.gripper_position);
        self.motion = Some(Motion::new(env, None));
        self.steps = 0;
        Ok(())
    }
    fn act(&mut self, env: &PickPlaceSim) -> Result<JointCommand> {
        self.steps += 1;
        let t = env.target();
        let cube_goal = [t[0], t[1], env.config().cube_half + self.carry_height];
        let goal = sub(cube_goal, self.hold_offset);
        let motion = self.motion.as_mut().expect("begin() first");
        motion.track(env, goal, GRIPPER_CLOSED)
    }
    fn status(&self, env: &PickPlaceSim) -> SkillStatus {
        if predicates::transported(env, &self.criteria) {
            SkillStatus::Success
        } else if predicates::dropped(env) {
            SkillStatus::Failure("dropped".into())
        } else if self.steps >= self.max_steps {
            SkillStatus::Failure("timeout".into())
        } else {
            SkillStatus::Running
        }
    }
}

// ---------------------------------------------------------------- Place
#[derive(Debug, Clone, Copy, PartialEq)]
enum PlacePhase {
    Lower,
    Release,
    Retreat,
    Settle,
}
pub struct ScriptedPlace {
    criteria: Criteria,
    motion: Option<Motion>,
    hold_offset: [f64; 3],
    phase: PlacePhase,
    phase_steps: usize,
    steps: usize,
    retreat_from: [f64; 3],
    pub max_steps: usize,
}
impl ScriptedPlace {
    pub fn new(criteria: Criteria) -> Self {
        Self {
            criteria,
            motion: None,
            hold_offset: [0.; 3],
            phase: PlacePhase::Lower,
            phase_steps: 0,
            steps: 0,
            retreat_from: [0.; 3],
            max_steps: 200,
        }
    }
    fn next(&mut self, phase: PlacePhase) {
        self.phase = phase;
        self.phase_steps = 0;
    }
}
impl Skill for ScriptedPlace {
    fn name(&self) -> &'static str {
        "place"
    }
    fn begin(&mut self, env: &PickPlaceSim) -> Result<()> {
        let s = env.state();
        self.hold_offset = sub(s.cube_position, s.gripper_position);
        self.motion = Some(Motion::new(env, None));
        self.phase = PlacePhase::Lower;
        self.phase_steps = 0;
        self.steps = 0;
        Ok(())
    }
    fn act(&mut self, env: &PickPlaceSim) -> Result<JointCommand> {
        self.steps += 1;
        self.phase_steps += 1;
        let t = env.target();
        // Release slightly above the table so the cube is not pushed into it.
        let cube_goal = [t[0], t[1], env.config().cube_half + 0.002];
        let goal = sub(cube_goal, self.hold_offset);
        let motion = self.motion.as_mut().expect("begin() first");
        Ok(match self.phase {
            PlacePhase::Lower => {
                motion.speed = 0.06;
                let c = motion.track(env, goal, GRIPPER_CLOSED)?;
                if motion.arrived(env, goal, 0.004) || self.phase_steps > 90 {
                    self.next(PlacePhase::Release);
                }
                c
            }
            PlacePhase::Release => {
                let c = motion.track(env, goal, GRIPPER_OPEN)?;
                if self.phase_steps >= 12 {
                    self.retreat_from = goal;
                    self.next(PlacePhase::Retreat);
                }
                c
            }
            PlacePhase::Retreat => {
                motion.speed = 0.10;
                let up = add(self.retreat_from, [0., 0., 0.06]);
                let c = motion.track(env, up, GRIPPER_OPEN)?;
                if motion.arrived(env, up, 0.01) || self.phase_steps > 45 {
                    self.next(PlacePhase::Settle);
                }
                c
            }
            PlacePhase::Settle => {
                let up = add(self.retreat_from, [0., 0., 0.06]);
                motion.track(env, up, GRIPPER_OPEN)?
            }
        })
    }
    fn status(&self, env: &PickPlaceSim) -> SkillStatus {
        if self.phase == PlacePhase::Settle && self.phase_steps >= 10 {
            if predicates::placed(env, &self.criteria) {
                SkillStatus::Success
            } else {
                SkillStatus::Failure("not resting inside tolerance".into())
            }
        } else if self.steps >= self.max_steps {
            SkillStatus::Failure("timeout".into())
        } else {
            SkillStatus::Running
        }
    }
}

// ------------------------------------------------------------ Sequencer
#[derive(Debug, Clone, Serialize)]
pub struct SkillReport {
    pub name: String,
    pub learned: bool,
    pub success: bool,
    pub steps: usize,
    pub reason: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct EpisodeReport {
    pub seed: u64,
    pub cube_initial: [f64; 3],
    pub target: [f64; 3],
    pub cube_final: [f64; 3],
    pub final_xy_error: f64,
    pub success: bool,
    pub skills: Vec<SkillReport>,
    pub control_steps: usize,
}

/// Run skills in order; each starts from the state its predecessor left.
/// The first failure ends the episode.
pub fn run_episode(
    env: &mut PickPlaceSim,
    skills: &mut [Box<dyn Skill>],
    seed: u64,
    mut on_step: impl FnMut(&PickPlaceSim, &str),
) -> Result<EpisodeReport> {
    env.reset(seed)?;
    let cube_initial = env.cube_pose().0;
    let mut reports = Vec::new();
    let mut total = 0;
    let mut success = true;
    for skill in skills.iter_mut() {
        skill.begin(env)?;
        let mut steps = 0;
        let status = loop {
            let status = skill.status(env);
            if status != SkillStatus::Running {
                break status;
            }
            let command = skill.act(env)?;
            env.step(&command, CONTROL_PERIOD)?;
            steps += 1;
            on_step(env, skill.name());
        };
        total += steps;
        let ok = status == SkillStatus::Success;
        reports.push(SkillReport {
            name: skill.name().into(),
            learned: skill.learned(),
            success: ok,
            steps,
            reason: match status {
                SkillStatus::Failure(r) => Some(r),
                _ => None,
            },
        });
        if !ok {
            success = false;
            break;
        }
    }
    let cube_final = env.cube_pose().0;
    let target = env.target();
    Ok(EpisodeReport {
        seed,
        cube_initial,
        target,
        cube_final,
        final_xy_error: xy_distance(cube_final, target),
        success,
        skills: reports,
        control_steps: total,
    })
}

pub fn scripted_skills(criteria: &Criteria) -> Vec<Box<dyn Skill>> {
    vec![
        Box::new(ScriptedGrasp::new(criteria.clone())),
        Box::new(ScriptedTransport::new(criteria.clone())),
        Box::new(ScriptedPlace::new(criteria.clone())),
    ]
}
