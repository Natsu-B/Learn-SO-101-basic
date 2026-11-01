//! Grasp-force control environment on the SO-101 (docs/FORCE_CONTROL.md §3).
//!
//! Separate from the pick-and-place environment: its own scene
//! (`scenes/so101_force_grasp.xml`) with a *compliant* object. The object is a
//! rigid core on a free joint whose four side faces are thin plates on
//! spring-damper slide joints, so the object's deformation is an explicit state
//! (`d`, the plate compression) with the explicit law `F = k d + c d'` up to a
//! travel limit. The jaw/plate contact keeps MuJoCo's soft-contact penetration,
//! which is a numerical artefact and is reported separately; the plates have a
//! fixed armature, so that penetration does not depend on the object mass.
//!
//! An episode starts like the scripted Grasp (approach, descend with the jaws
//! open) and freezes the arm at the start of the Close phase; afterwards only
//! the gripper is commanded. The scripted lift reuses `ScriptedGrasp`.
//! Reimplemented from the paper's description (Lach et al., IROS 2024,
//! arXiv:2311.07245); no code from github.com/llach/learning_fc is copied.
use crate::pick_place::{GRIPPER, IkTarget, PickPlaceConfig, PickPlaceSim};
use crate::skills::{CONTROL_PERIOD, Criteria, ScriptedGrasp, Skill};
use crate::{model_access, presets};
use robotics_core::{Error, JointCommand, Result};
use rusty_mujoco::{self as mj, ObjectId, joint, obj};
use serde::{Deserialize, Serialize};

pub const SCENE_TEMPLATE: &str = include_str!("../../../scenes/so101_force_grasp.xml");
/// Half thickness of a compliant face plate and its lateral/vertical extent
/// relative to the core (matches the scene's `skin` class).
const PLATE_HALF: f64 = 0.0025;
const PLATE_SPAN: f64 = 0.9;
const PLATE_HALF_HEIGHT: f64 = 0.0185;
pub const OBJECT_HALF_HEIGHT: f64 = 0.02;
const BASE_HALF_HEIGHT: f64 = 0.0012;
/// Moving mass of one face in its joint (plate mass + armature, kg).
pub const PLATE_MASS: f64 = 0.102;
/// Gap between the fixed jaw and the object face after the scripted descent
/// with `jaw_clearance = 0`: the measured gap (push of the object by the
/// moving jaw before the fixed jaw is reached) is 1 mm below the requested
/// clearance (`force_grasp_probe calibrate`). With the default ranges (width
/// <= 30 mm, gap >= 4 mm) the jaws do not hit the object while descending;
/// wider objects or smaller gaps can (docs/FORCE_CONTROL.md §3).
pub const FIXED_JAW_BASE_GAP: f64 = -0.001;
const SKINS: [&str; 4] = ["skin_px", "skin_nx", "skin_py", "skin_ny"];

/// How `ForceGraspSim::reset` brings the open gripper around the object.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Positioning {
    /// Run the scripted Grasp approach and descent (default; docs §3–4).
    #[default]
    Scripted,
    /// Solve the same descent target by IK once and place the arm there at
    /// rest (reset-time state, ~20x faster; used for RL training).
    Teleport,
}

/// Physical properties of one compliant object. Hidden from the policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompliantObject {
    /// Outer width of the (square) object (m).
    pub width: f64,
    /// Fixed-jaw clearance to the object face when closing starts (m). The
    /// SO-101 counterpart of the paper's object offset `o_y`.
    pub gap: f64,
    /// Face spring stiffness (N/m) and damping (N s/m).
    pub stiffness: f64,
    pub damping: f64,
    /// Maximum face compression before the core is reached (m); at most
    /// `width / 2 − 8 mm` so that the core keeps a half size of 3 mm.
    pub travel: f64,
    /// Jaw/face sliding friction.
    pub friction: f64,
    /// Core mass (kg); the four plates add 8 g.
    pub mass: f64,
}
impl CompliantObject {
    pub fn nominal() -> Self {
        Self {
            width: 0.03,
            gap: 0.004,
            stiffness: 1000.,
            damping: 2.,
            travel: 0.004,
            friction: 1.0,
            mass: 0.03,
        }
    }
    pub fn validate(&self) -> Result<()> {
        let ok = |v: f64, lo: f64, hi: f64| v.is_finite() && (lo..=hi).contains(&v);
        if !(ok(self.width, 0.012, 0.045)
            && ok(self.gap, 0.0, 0.015)
            && ok(self.stiffness, 10., 1e5)
            && ok(self.damping, 0., 1e3)
            && ok(
                self.travel,
                1e-4,
                0.5 * self.width - 2. * PLATE_HALF - 0.003 + 1e-12,
            )
            && ok(self.friction, 0.01, 3.)
            && ok(self.mass, 1e-3, 1.))
        {
            return Err(Error::Input(format!(
                "compliant object out of range: {self:?}"
            )));
        }
        Ok(())
    }
    /// Largest steady face force before bottoming out (N).
    pub fn max_spring_force(&self) -> f64 {
        self.stiffness * self.travel
    }
    /// Goal-force interval the object can bear: [f_min, min(0.8 k·travel,
    /// f_cap)], as the paper samples f_goal below f_max(kappa).
    pub fn goal_range(&self, f_min: f64, f_cap: f64) -> [f64; 2] {
        [f_min, (0.8 * self.max_spring_force()).min(f_cap).max(f_min)]
    }
}

/// Uniform sampling ranges. `kappa` interpolates log-stiffness between
/// `stiffness[0]` (hard, kappa = 0) and `stiffness[1]` (soft), as the paper's
/// unified stiffness factor does for its solver width and force scale.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForceGraspRanges {
    pub width: [f64; 2],
    pub gap: [f64; 2],
    pub kappa: [f64; 2],
    pub stiffness: [f64; 2],
    /// Face damping ratio relative to critical damping of the plate.
    pub damping_ratio: [f64; 2],
    pub travel: [f64; 2],
    pub friction: [f64; 2],
    pub mass: [f64; 2],
    /// Gripper joint damping scale (the SO-101 counterpart of the paper's
    /// actuator bias b2 randomization). Default: nominal.
    pub gripper_damping_scale: [f64; 2],
}
impl Default for ForceGraspRanges {
    fn default() -> Self {
        Self {
            width: [0.022, 0.03],
            gap: [0.004, 0.008],
            kappa: [0., 1.],
            stiffness: [5000., 200.],
            damping_ratio: [0.3, 1.0],
            travel: [0.003, 0.006],
            friction: [0.3, 1.5],
            mass: [0.01, 0.1],
            gripper_damping_scale: [1., 1.],
        }
    }
}
impl ForceGraspRanges {
    pub fn sample_gripper_damping(&self, seed: u64) -> f64 {
        let mut rng = seed ^ 0xda4b_0000_5ca1_e000;
        let [lo, hi] = self.gripper_damping_scale;
        lo + (hi - lo) * uniform(&mut rng)
    }
    pub fn stiffness_at(&self, kappa: f64) -> f64 {
        let [hard, soft] = self.stiffness;
        (hard.ln() + kappa * (soft.ln() - hard.ln())).exp()
    }
    pub fn sample(&self, seed: u64) -> CompliantObject {
        let mut rng = seed ^ 0xf0ce_9a1c_0b1e_c700;
        let mut u = |r: [f64; 2]| r[0] + (r[1] - r[0]) * uniform(&mut rng);
        let width = u(self.width);
        let gap = u(self.gap);
        let kappa = u(self.kappa);
        let zeta = u(self.damping_ratio);
        let travel = u(self.travel);
        let friction = u(self.friction);
        let mass = u(self.mass);
        let stiffness = self.stiffness_at(kappa);
        CompliantObject {
            width,
            gap,
            stiffness,
            damping: 2. * zeta * (stiffness * PLATE_MASS).sqrt(),
            travel: travel.min(0.5 * width - 2. * PLATE_HALF - 0.003),
            friction,
            mass,
        }
    }
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1_u64 << 53) as f64)
}

/// Jaw penetration above which the contact is treated as geometrically
/// invalid (a plate tunnelling into a jaw pad) rather than a measurement.
pub const PENETRATION_LIMIT: f64 = 0.001;

/// One measurement. Index 0 is the fixed jaw side, 1 the moving jaw side.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct ForceReading {
    pub time: f64,
    pub gripper_q: f64,
    pub gripper_qvel: f64,
    pub gripper_command: f64,
    pub actuator_torque: f64,
    pub torque_limit: f64,
    /// Jaw normal force on the compliant faces (N) — the "tactile" signal.
    pub jaw_force: [f64; 2],
    /// Jaw normal force on anything else of the object (core, other faces).
    pub rigid_force: [f64; 2],
    /// Face compression of the grasped faces (m) and its spring-damper force.
    pub deformation: [f64; 2],
    pub spring_force: [f64; 2],
    /// Grasped face at its travel limit (the object is crushed to its core;
    /// further force goes through the stiff joint limit, not the spring).
    pub bottomed: [bool; 2],
    /// Deepest jaw/object contact penetration (m, numerical).
    pub penetration: f64,
    pub table_force: f64,
    pub object_position: [f64; 3],
    pub object_velocity: [f64; 3],
    /// Object velocity along the jaw closing direction (m/s).
    pub object_closing_velocity: f64,
    /// Horizontal object displacement since the episode start (m).
    pub object_displacement: f64,
    /// Core shift towards the fixed jaw minus the fixed-side compression (m):
    /// how far the face touching the fixed jaw moved (push across the gap,
    /// then slip). Squeezing alone moves the core but not this face.
    pub fixed_face_shift: f64,
}

pub struct ForceGraspSim {
    pub pp: PickPlaceSim,
    ranges: ForceGraspRanges,
    core_body: ObjectId<obj::Body>,
    core_geom: ObjectId<obj::Geom>,
    column_geom: ObjectId<obj::Geom>,
    skin_bodies: [ObjectId<obj::Body>; 4],
    skin_geoms: [ObjectId<obj::Geom>; 4],
    skin_joints: [ObjectId<joint::Slide>; 4],
    fixed_jaw_body: ObjectId<obj::Body>,
    moving_jaw_body: ObjectId<obj::Body>,
    nominal_core: [f64; 6],
    object: CompliantObject,
    /// Skin index on the fixed and moving jaw side.
    sides: [usize; 2],
    arm_command: JointCommand,
    closing: [f64; 3],
    start: [f64; 3],
    /// Object displacement while the arm was positioned (m).
    pub positioning_displacement: f64,
    /// Gripper damping scale of this episode.
    pub gripper_damping_scale: f64,
    pub positioning: Positioning,
    /// Teleport positioning: IK target, closing constraint and solution, for
    /// `shift_arm` (arm-assisted experiment, docs §10.5).
    pinch_target: Option<([f64; 3], [f64; 3], Vec<f64>)>,
    arm_shift: f64,
}

impl ForceGraspSim {
    pub fn new(ranges: ForceGraspRanges) -> Result<Self> {
        let spec = presets::robot("so101")?;
        let config = PickPlaceConfig {
            cube_half: OBJECT_HALF_HEIGHT,
            ..PickPlaceConfig::default()
        };
        let pp = PickPlaceSim::from_scene(spec, config, SCENE_TEMPLATE)?;
        let m = &pp.sim.model;
        let missing = |w: &str| Error::Model(format!("force-grasp scene is missing {w}"));
        let body = |n: &str| m.object_id::<obj::Body>(n).ok_or_else(|| missing(n));
        let geom = |n: &str| m.object_id::<obj::Geom>(n).ok_or_else(|| missing(n));
        let slide = |n: &str| m.object_id::<joint::Slide>(n).ok_or_else(|| missing(n));
        let skin_bodies = [
            body(SKINS[0])?,
            body(SKINS[1])?,
            body(SKINS[2])?,
            body(SKINS[3])?,
        ];
        let skin_geoms = [
            geom(SKINS[0])?,
            geom(SKINS[1])?,
            geom(SKINS[2])?,
            geom(SKINS[3])?,
        ];
        let skin_joints = [
            slide(SKINS[0])?,
            slide(SKINS[1])?,
            slide(SKINS[2])?,
            slide(SKINS[3])?,
        ];
        let core_body = body("cube")?;
        let core_geom = geom("cube_geom")?;
        let column_geom = geom("cube_column")?;
        let fixed_jaw_body = body("gripper")?;
        let moving_jaw_body = body("moving_jaw_so101_v1")?;
        let nominal_core = model_access::object_parameters(m, core_body, core_geom)?;
        let mut this = Self {
            pp,
            ranges,
            core_body,
            core_geom,
            column_geom,
            skin_bodies,
            skin_geoms,
            skin_joints,
            fixed_jaw_body,
            moving_jaw_body,
            nominal_core,
            object: CompliantObject::nominal(),
            sides: [0, 1],
            arm_command: JointCommand::default(),
            closing: [1., 0., 0.],
            start: [0.; 3],
            positioning_displacement: 0.,
            gripper_damping_scale: 1.,
            positioning: Positioning::Scripted,
            pinch_target: None,
            arm_shift: 0.,
        };
        this.apply_object(CompliantObject::nominal())?;
        Ok(this)
    }
    pub fn ranges(&self) -> &ForceGraspRanges {
        &self.ranges
    }
    pub fn set_ranges(&mut self, ranges: ForceGraspRanges) {
        self.ranges = ranges;
    }
    pub fn object(&self) -> CompliantObject {
        self.object
    }
    pub fn skin_geoms(&self) -> &[ObjectId<obj::Geom>; 4] {
        &self.skin_geoms
    }

    /// Write the object's geometry, springs, friction and mass into the model.
    /// Reset-time only: the caller resets mjData afterwards.
    fn apply_object(&mut self, o: CompliantObject) -> Result<()> {
        o.validate()?;
        let m = &mut self.pp.sim.model;
        // The core sits `travel` inside the plates' inner faces, so a jaw can
        // reach it only after the full compression.
        let inner = 0.5 * o.width - 2. * PLATE_HALF;
        let core = inner - o.travel;
        model_access::set_box(m, self.core_geom, [inner, inner, BASE_HALF_HEIGHT])?;
        model_access::set_box(m, self.column_geom, [core, core, OBJECT_HALF_HEIGHT])?;
        let plate = [PLATE_HALF, PLATE_SPAN * inner, PLATE_HALF_HEIGHT];
        let offset = inner + PLATE_HALF;
        let positions = [[offset, 0.], [-offset, 0.], [0., offset], [0., -offset]];
        for (i, [x, y]) in positions.into_iter().enumerate() {
            model_access::set_box(m, self.skin_geoms[i], plate)?;
            model_access::set_body_pos(m, self.skin_bodies[i], [x, y, 0.])?;
            model_access::set_slide(
                m,
                self.skin_joints[i].index(),
                o.stiffness,
                o.damping,
                o.travel,
            )?;
            let mut c = model_access::geom_contact(m, self.skin_geoms[i])?;
            c[0] = o.friction;
            model_access::set_geom_contact(m, self.skin_geoms[i], &c)?;
        }
        // Core mass with the inertia of a solid box of the outer size;
        // write_object also runs mj_setConst.
        let mut core_values = self.nominal_core;
        let (a, h) = (inner, OBJECT_HALF_HEIGHT);
        core_values[0] = o.mass;
        core_values[1] = o.mass * (a * a + h * h) / 3.;
        core_values[2] = core_values[1];
        core_values[3] = o.mass * 2. * a * a / 3.;
        let (model, data) = (&mut self.pp.sim.model, &mut self.pp.sim.data);
        model_access::set_object_parameters(
            model,
            data,
            self.core_body,
            self.core_geom,
            &core_values,
        )?;
        self.object = o;
        Ok(())
    }

    /// Reset with a sampled object (or `object`), position the open gripper
    /// around it with the scripted Grasp and freeze the arm.
    pub fn reset(&mut self, seed: u64, object: Option<CompliantObject>) -> Result<()> {
        let o = object.unwrap_or_else(|| self.ranges.sample(seed));
        self.apply_object(o)?;
        let scale = self.ranges.sample_gripper_damping(seed);
        let mut p = self.pp.sim.parameters().clone();
        p.joint_damping_scale[GRIPPER] = scale;
        self.pp.sim.set_parameter_center(p)?;
        self.gripper_damping_scale = scale;
        self.pp.reset(seed)?;
        let start = self.pp.cube_pose().0;
        // The scripted descent is calibrated for a 25 mm cube: shift the fixed
        // jaw by the extra half width plus the requested gap.
        let clearance = (0.5 * o.width - 0.0125) + o.gap - FIXED_JAW_BASE_GAP;
        self.pinch_target = None;
        self.arm_shift = 0.;
        let command = match self.positioning {
            Positioning::Scripted => self.position_scripted(clearance)?,
            Positioning::Teleport => self.position_teleport(start, clearance)?,
        };
        self.arm_command = command;
        self.closing = self.pp.closing_direction();
        let core = self.pp.cube_pose().0;
        let gx = self.geom_positions();
        let along = |i: usize| {
            (0..3)
                .map(|k| (gx[i][k] - core[k]) * self.closing[k])
                .sum::<f64>()
        };
        let order = |a: &usize, b: &usize| along(*a).total_cmp(&along(*b));
        self.sides = [
            (0..4).min_by(order).expect("four skins"),
            (0..4).max_by(order).expect("four skins"),
        ];
        self.start = core;
        self.positioning_displacement = (core[0] - start[0]).hypot(core[1] - start[1]);
        Ok(())
    }
    fn position_scripted(&mut self, clearance: f64) -> Result<JointCommand> {
        let mut g = ScriptedGrasp::new(Criteria::default());
        g.jaw_clearance = clearance;
        g.begin(&self.pp)?;
        let mut command = g.act(&self.pp)?;
        while g.phase() != Some("close") {
            self.pp.step(&command, CONTROL_PERIOD)?;
            command = g.act(&self.pp)?;
        }
        // The descent ends within 3 mm of its target; keep tracking it through
        // the Close phase with the jaws open so the frozen pose is converged.
        let open = self.pp.sim.applied_command().position[GRIPPER];
        while g.phase() == Some("close") {
            command.position[GRIPPER] = open;
            self.pp.step(&command, CONTROL_PERIOD)?;
            command = g.act(&self.pp)?;
        }
        let mut command = self.pp.sim.applied_command().clone();
        command.position[GRIPPER] = open;
        for _ in 0..10 {
            self.pp.step(&command, CONTROL_PERIOD)?;
        }
        Ok(command)
    }
    /// Same targets as the scripted Grasp (approach closing direction, descent
    /// point shifted by `clearance` away from the fixed jaw), solved once.
    fn position_teleport(&mut self, cube: [f64; 3], clearance: f64) -> Result<JointCommand> {
        let above = [cube[0], cube[1], cube[2] + 0.06];
        let closing = crate::skills::grasp_closing(&self.pp, above)?;
        let q0 = self.pp.sim.applied_command().position.clone();
        let (q_above, _) = self
            .pp
            .solve_ik(&q0, &IkTarget::top_down(above, Some(closing)), 200)?;
        let z = self.pp.fk_closing(&q_above)?;
        let h = z[0].hypot(z[1]).max(1e-9);
        let target = [
            cube[0] - clearance * z[0] / h,
            cube[1] - clearance * z[1] / h,
            cube[2],
        ];
        let (mut q, _) =
            self.pp
                .solve_ik(&q_above, &IkTarget::top_down(target, Some(closing)), 200)?;
        self.pinch_target = Some((target, closing, q.clone()));
        q[GRIPPER] = crate::skills::GRIPPER_OPEN;
        self.pp.sim.teleport(&q)?;
        let command = JointCommand { position: q };
        for _ in 0..10 {
            self.pp.step(&command, CONTROL_PERIOD)?;
        }
        Ok(command)
    }
    fn geom_positions(&self) -> [[f64; 3]; 4] {
        let m = &self.pp.sim.model;
        // SAFETY: matching model/data, read-only.
        let gx = unsafe { self.pp.sim.data.geom_xpos(m) };
        std::array::from_fn(|i| {
            let g = self.skin_geoms[i].index();
            [gx[3 * g], gx[3 * g + 1], gx[3 * g + 2]]
        })
    }

    pub fn gripper_bounds(&self) -> [f64; 2] {
        self.pp.sim.action_bounds()[GRIPPER]
    }
    /// One control period with the arm frozen and the given gripper target.
    /// Returns the extremes over its physics steps.
    pub fn step(&mut self, gripper: f64) -> Result<StepStats> {
        if !gripper.is_finite() {
            return Err(Error::Input("non-finite gripper command".into()));
        }
        let [lo, hi] = self.gripper_bounds();
        let mut c = self.arm_command.clone();
        c.position[GRIPPER] = gripper.clamp(lo, hi);
        let probe = self.probe();
        let travel = self.object.travel;
        let gripper_actuator = self.pp.sim.actuator_ids()[GRIPPER];
        let mut st = StepStats {
            min_jaw_force: [f64::INFINITY; 2],
            ..StepStats::default()
        };
        self.pp
            .sim
            .step_control_observed(&c, CONTROL_PERIOD, |sim| {
                let scan = probe.scan(&sim.model, &sim.data);
                st.physics_steps += 1;
                for k in 0..2 {
                    st.max_jaw_force[k] = st.max_jaw_force[k].max(scan.jaw_force[k]);
                    st.min_jaw_force[k] = st.min_jaw_force[k].min(scan.jaw_force[k]);
                    st.max_rigid_force[k] = st.max_rigid_force[k].max(scan.rigid_force[k]);
                    st.max_deformation[k] = st.max_deformation[k].max(scan.deformation[k]);
                    st.bottomed[k] |= scan.deformation[k] >= travel - BOTTOM_TOLERANCE;
                }
                st.max_penetration = st.max_penetration.max(scan.penetration);
                // SAFETY: matching model/data, read-only slice of nu entries.
                let tau = unsafe { sim.data.actuator_force(&sim.model) }[gripper_actuator];
                st.max_actuator_torque = st.max_actuator_torque.max(tau.abs());
            })?;
        let (p, _) = self.pp.cube_pose();
        if !p.iter().all(|v| v.is_finite()) {
            return Err(Error::Unstable);
        }
        Ok(st)
    }
    /// Maximum `shift_arm` displacement (m).
    pub const MAX_ARM_SHIFT: f64 = 0.012;
    /// Arm-assisted experiment (not the paper): move the frozen pinch target
    /// horizontally along the closing direction by `shift` metres from its
    /// reset position (positive: from the fixed towards the moving jaw, so
    /// the fixed jaw approaches the object and the moving jaw backs off).
    /// Re-solves the arm IK; the gripper command is unchanged. Teleport
    /// positioning only.
    pub fn shift_arm(&mut self, shift: f64) -> Result<()> {
        let Some((target, closing, q0)) = self.pinch_target.clone() else {
            return Err(Error::Input("shift_arm needs teleport positioning".into()));
        };
        if !(shift.is_finite() && shift.abs() <= Self::MAX_ARM_SHIFT) {
            return Err(Error::Input("arm shift out of range".into()));
        }
        let z = self.closing;
        let h = z[0].hypot(z[1]).max(1e-9);
        let goal = [
            target[0] + shift * z[0] / h,
            target[1] + shift * z[1] / h,
            target[2],
        ];
        let (q, _) = self
            .pp
            .solve_ik(&q0, &IkTarget::top_down(goal, Some(closing)), 100)?;
        for (i, v) in q.iter().enumerate() {
            if i != GRIPPER {
                self.arm_command.position[i] = *v;
            }
        }
        self.arm_shift = shift;
        Ok(())
    }
    pub fn arm_shift(&self) -> f64 {
        self.arm_shift
    }
    /// Arm command frozen at reset (gripper entry is the opening command).
    pub fn arm_command(&self) -> &JointCommand {
        &self.arm_command
    }

    /// Every jaw/table contact of the object: (object geom, other geom, normal
    /// force N, distance m). Diagnostic only.
    pub fn object_contacts(&self) -> Vec<(String, String, f64, f64)> {
        let m = &self.pp.sim.model;
        let mut out = Vec::new();
        for g in self
            .skin_geoms
            .iter()
            .chain([&self.core_geom, &self.column_geom])
        {
            for r in model_access::geom_contacts(m, &self.pp.sim.data, *g) {
                out.push((
                    model_access::geom_name(m, g.index()),
                    model_access::geom_name(m, r[0] as usize),
                    r[13],
                    r[2],
                ));
            }
        }
        out
    }

    /// Body ids of the fixed and moving jaw.
    pub fn jaw_body_ids(&self) -> [usize; 2] {
        [self.fixed_jaw_body.index(), self.moving_jaw_body.index()]
    }
    /// Robot geoms touching the table: (geom, normal force N). Diagnostic only.
    pub fn robot_table_contacts(&self) -> Vec<(String, f64)> {
        let m = &self.pp.sim.model;
        model_access::geom_contacts(m, &self.pp.sim.data, self.pp.table_geom())
            .into_iter()
            .filter(|r| {
                let g = r[0] as usize;
                g != self.core_geom.index()
                    && g != self.column_geom.index()
                    && !self.skin_geoms.iter().any(|s| s.index() == g)
            })
            .map(|r| {
                (
                    format!(
                        "{}#{}@body{}",
                        model_access::geom_name(m, r[0] as usize),
                        r[0],
                        r[1]
                    ),
                    r[13],
                )
            })
            .collect()
    }

    fn probe(&self) -> Probe {
        Probe {
            core_geom: self.core_geom,
            column_geom: self.column_geom,
            skin_geoms: self.skin_geoms,
            skin_joints: self.skin_joints,
            fixed_jaw_body: self.fixed_jaw_body,
            moving_jaw_body: self.moving_jaw_body,
            table_geom: self.pp.table_geom(),
            sides: self.sides,
        }
    }

    pub fn reading(&self) -> Result<ForceReading> {
        let m = &self.pp.sim.model;
        let d = &self.pp.sim.data;
        let scan = self.probe().scan(m, d);
        let mut r = ForceReading {
            jaw_force: scan.jaw_force,
            rigid_force: scan.rigid_force,
            table_force: scan.table_force,
            penetration: scan.penetration,
            deformation: scan.deformation,
            ..ForceReading::default()
        };
        for side in 0..2 {
            r.bottomed[side] = scan.deformation[side] >= self.object.travel - BOTTOM_TOLERANCE;
            r.spring_force[side] = self.object.stiffness * scan.deformation[side]
                + self.object.damping * scan.deformation_rate[side];
        }
        let js = self.pp.sim.state();
        let f = self.pp.grip_forces()?;
        r.time = self.pp.sim.time();
        r.gripper_q = js.position[GRIPPER];
        r.gripper_qvel = js.velocity[GRIPPER];
        r.gripper_command = self.pp.sim.applied_command().position[GRIPPER];
        r.actuator_torque = f.gripper_actuator_force;
        r.torque_limit = f.gripper_force_range[1];
        let (p, _) = self.pp.cube_pose();
        let v = self.pp.cube_velocity();
        r.object_position = p;
        r.object_velocity = [v[0], v[1], v[2]];
        r.object_closing_velocity = (0..3).map(|k| v[k] * self.closing[k]).sum();
        r.object_displacement = (p[0] - self.start[0]).hypot(p[1] - self.start[1]);
        let towards_fixed: f64 = -(0..3)
            .map(|k| (p[k] - self.start[k]) * self.closing[k])
            .sum::<f64>();
        r.fixed_face_shift = towards_fixed - r.deformation[0];
        Ok(r)
    }
}

const BOTTOM_TOLERANCE: f64 = 1e-5;

/// Object/jaw element ids for contact scans (copyable, so that a scan can run
/// inside the per-physics-step hook while the simulator is borrowed).
#[derive(Clone, Copy)]
struct Probe {
    core_geom: ObjectId<obj::Geom>,
    column_geom: ObjectId<obj::Geom>,
    skin_geoms: [ObjectId<obj::Geom>; 4],
    skin_joints: [ObjectId<joint::Slide>; 4],
    fixed_jaw_body: ObjectId<obj::Body>,
    moving_jaw_body: ObjectId<obj::Body>,
    table_geom: ObjectId<obj::Geom>,
    sides: [usize; 2],
}
#[derive(Default)]
struct Scan {
    jaw_force: [f64; 2],
    rigid_force: [f64; 2],
    table_force: f64,
    penetration: f64,
    deformation: [f64; 2],
    deformation_rate: [f64; 2],
}
impl Probe {
    fn scan(&self, m: &mj::mjModel, d: &mj::mjData) -> Scan {
        let mut r = Scan::default();
        let is_object = |g: ObjectId<obj::Geom>| {
            g == self.core_geom || g == self.column_geom || self.skin_geoms.contains(&g)
        };
        for (i, c) in d.contact().iter().enumerate() {
            let (Some(a), Some(b)) = c.geom() else {
                continue;
            };
            let (own, other) = if is_object(a) {
                (a, b)
            } else if is_object(b) {
                (b, a)
            } else {
                continue;
            };
            let normal = mj::mj_contactForce(m, d, i)[0];
            if other == self.table_geom {
                r.table_force += normal;
                continue;
            }
            let body = m.geom_bodyid(other);
            let side = if body == self.fixed_jaw_body {
                0
            } else if body == self.moving_jaw_body {
                1
            } else {
                continue;
            };
            r.penetration = r.penetration.max(-c.dist());
            if own == self.skin_geoms[self.sides[side]] {
                r.jaw_force[side] += normal;
            } else {
                r.rigid_force[side] += normal;
            }
        }
        for (side, &s) in self.sides.iter().enumerate() {
            r.deformation[side] = d.qpos(self.skin_joints[s], m)[0];
            r.deformation_rate[side] = d.qvel(self.skin_joints[s], m)[0];
        }
        r
    }
}

/// Extremes over the physics steps (5 ms) of one control period, so that
/// impact peaks between control samples are not missed. Index 0 = fixed jaw.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct StepStats {
    pub physics_steps: usize,
    /// Largest normal force on the grasped faces and on anything else.
    pub max_jaw_force: [f64; 2],
    pub max_rigid_force: [f64; 2],
    /// Smallest face force (contact lost within the period if ~0).
    pub min_jaw_force: [f64; 2],
    pub max_deformation: [f64; 2],
    pub bottomed: [bool; 2],
    pub max_penetration: f64,
    pub max_actuator_torque: f64,
}

/// Hand-designed grasp force controller after the structure of Lach et al.
/// (IROS 2022) as used as the paper's baseline: close at constant speed until
/// both jaws touch, then a PI law on the force error converted to an angle
/// through an assumed stiffness. Output: gripper position delta (rad).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForcePi {
    pub kp: f64,
    pub ki: f64,
    /// Assumed grasp stiffness (N/rad) converting force error to angle.
    pub stiffness: f64,
    /// Closing speed before contact (rad/s, positive closes).
    pub closing_speed: f64,
    /// Contact threshold (N).
    pub threshold: f64,
    integral: f64,
    contact: [bool; 2],
}
impl Default for ForcePi {
    /// Gains from the small sweep in docs/FORCE_CONTROL.md §4 (closing at
    /// 0.6 rad/s, an assumed 100 N/rad grasp stiffness, 0.05 N threshold).
    fn default() -> Self {
        Self::new(0.5, 2.0, 100., 0.6, 0.05)
    }
}
impl ForcePi {
    pub fn new(kp: f64, ki: f64, stiffness: f64, closing_speed: f64, threshold: f64) -> Self {
        Self {
            kp,
            ki,
            stiffness,
            closing_speed,
            threshold,
            integral: 0.,
            contact: [false; 2],
        }
    }
    pub fn reset(&mut self) {
        self.integral = 0.;
        self.contact = [false; 2];
    }
    /// Gripper delta for this period; `forces` = [fixed, moving] in N. The
    /// SO-101 gripper closes towards negative angles.
    pub fn act(&mut self, forces: [f64; 2], goal: f64, dt: f64) -> f64 {
        for (c, f) in self.contact.iter_mut().zip(forces) {
            *c |= f > self.threshold;
        }
        if !(self.contact[0] && self.contact[1]) {
            return -self.closing_speed * dt;
        }
        let error = (goal - 0.5 * (forces[0] + forces[1])) / self.stiffness;
        self.integral += error * dt;
        -(self.kp * error + self.ki * self.integral)
    }
}
