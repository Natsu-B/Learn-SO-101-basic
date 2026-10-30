//! SO-101 cube pick-and-place scene: dynamic cube, mocap goal marker, contact
//! queries and a position+axis IK. The cube is moved only by contact physics;
//! this module writes its pose solely at reset.
use crate::{Simulator, presets};
use robotics_core::{Error, JointCommand, Result};
use rusty_mujoco::{self as mj, ObjectId, joint, obj};

pub const SCENE_TEMPLATE: &str = include_str!("../../../scenes/so101_pick_place.xml");
/// Arm actuators in API order (gripper excluded).
pub const ARM: [usize; 5] = [0, 1, 2, 3, 4];
pub const GRIPPER: usize = 5;

#[derive(Debug, Clone)]
pub struct PickPlaceConfig {
    /// Half edge of the cube in the scene XML (metres).
    pub cube_half: f64,
    /// Polar sampling region (radius from base z-axis, azimuth) for cube and goal.
    pub radius_range: [f64; 2],
    pub azimuth_range: [f64; 2],
    /// Uniform cube yaw range (radians).
    pub yaw_range: [f64; 2],
    pub min_separation: f64,
    /// XY tolerance for a successful placement.
    pub place_tolerance: f64,
    /// Height above rest for a successful grasp/lift.
    pub lift_height: f64,
}
impl Default for PickPlaceConfig {
    fn default() -> Self {
        Self {
            cube_half: 0.0125,
            radius_range: [0.15, 0.25],
            azimuth_range: [-0.8, 0.8],
            yaw_range: [0.0, 0.0],
            min_separation: 0.08,
            place_tolerance: 0.02,
            lift_height: 0.03,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Contacts {
    pub fixed_jaw: bool,
    pub moving_jaw: bool,
    pub table: bool,
    /// Cube touching any other robot link (not a jaw).
    pub other_robot: bool,
}
impl Contacts {
    /// Both jaws in contact: the cube is pinched by the gripper.
    pub fn pinched(&self) -> bool {
        self.fixed_jaw && self.moving_jaw
    }
}

#[derive(Debug, Clone)]
pub struct PickPlaceState {
    pub joint_position: Vec<f64>,
    pub joint_velocity: Vec<f64>,
    pub gripper_position: [f64; 3],
    /// Gripper approach axis (unit, world frame).
    pub gripper_axis: [f64; 3],
    pub cube_position: [f64; 3],
    pub cube_quat: [f64; 4],
    pub cube_linear_velocity: [f64; 3],
    pub target_position: [f64; 3],
    pub contacts: Contacts,
    pub time: f64,
}

pub struct PickPlaceSim {
    pub sim: Simulator,
    config: PickPlaceConfig,
    cube_joint: ObjectId<joint::Free>,
    cube_geom: ObjectId<obj::Geom>,
    table_geom: ObjectId<obj::Geom>,
    target_body: ObjectId<obj::Body>,
    fixed_jaw_body: ObjectId<obj::Body>,
    moving_jaw_body: ObjectId<obj::Body>,
    grip_site: ObjectId<obj::Site>,
    target: [f64; 3],
    /// Grasp centre in the gripperframe site's local frame (metres).
    pub grasp_offset: [f64; 3],
    /// Local site axis that points from the wrist towards the fingertips.
    pub approach_axis_local: [f64; 3],
    /// Jaw reference points in the gripperframe site frame at reset:
    /// fixed-jaw tip, fixed-jaw middle pad, moving-jaw tip.
    pub jaw_local: [[f64; 3]; 3],
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1_u64 << 53) as f64)
}
fn lerp(r: [f64; 2], u: f64) -> f64 {
    r[0] + (r[1] - r[0]) * u
}
pub(crate) fn mat_vec(m: &[f64], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|r| (0..3).map(|c| m[3 * r + c] * v[c]).sum())
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Solve (A) x = b for small dense symmetric positive definite A (Gaussian elimination).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        a.swap(k, p);
        b.swap(k, p);
        let d = a[k][k];
        if d.abs() < 1e-15 {
            return vec![0.; n];
        }
        for i in k + 1..n {
            let f = a[i][k] / d;
            let pivot = a[k].clone();
            for (v, p) in a[i][k..].iter_mut().zip(&pivot[k..]) {
                *v -= f * p;
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.; n];
    for k in (0..n).rev() {
        let s: f64 = (k + 1..n).map(|j| a[k][j] * x[j]).sum();
        x[k] = (b[k] - s) / a[k][k];
    }
    x
}

impl PickPlaceSim {
    pub fn new(config: PickPlaceConfig) -> Result<Self> {
        let spec = presets::robot("so101")?;
        let sim = Simulator::from_spec_with_scene(spec, SCENE_TEMPLATE)?;
        let m = &sim.model;
        let missing = |what: &str| Error::Model(format!("pick-place scene is missing {what}"));
        let cube_joint = m
            .object_id::<joint::Free>("cube_joint")
            .ok_or_else(|| missing("cube_joint"))?;
        let cube_geom = m
            .object_id::<obj::Geom>("cube_geom")
            .ok_or_else(|| missing("cube_geom"))?;
        let table_geom = m
            .object_id::<obj::Geom>("table")
            .ok_or_else(|| missing("table"))?;
        let target_body = m
            .object_id::<obj::Body>("place_target")
            .ok_or_else(|| missing("place_target"))?;
        let fixed_jaw_body = m
            .object_id::<obj::Body>("gripper")
            .ok_or_else(|| missing("gripper body"))?;
        let moving_jaw_body = m
            .object_id::<obj::Body>("moving_jaw_so101_v1")
            .ok_or_else(|| missing("moving jaw body"))?;
        let grip_site = m
            .object_id::<obj::Site>("gripperframe")
            .ok_or_else(|| missing("gripperframe"))?;
        let mut this = Self {
            sim,
            config,
            cube_joint,
            cube_geom,
            table_geom,
            target_body,
            fixed_jaw_body,
            moving_jaw_body,
            grip_site,
            target: [0.2, 0.12, 0.0],
            // Measured with examples/rust/pick_place_probe.rs: the fixed-jaw tip is
            // at site (+2.9, 0.2, -20.1) mm; a 25 mm cube pressed against the slanted
            // fixed pad, 9 mm behind the tip, has its centre at (-6, 0.2, -8) mm.
            grasp_offset: [-0.006, 0.0002, -0.008],
            approach_axis_local: [1.0, 0.0, 0.0],
            jaw_local: [[0.; 3]; 3],
        };
        this.calibrate_frame()?;
        Ok(this)
    }
    pub fn config(&self) -> &PickPlaceConfig {
        &self.config
    }

    /// Derive the approach axis and grasp centre from the jaw-tip geoms
    /// instead of assuming that gripperframe is the pinch centre.
    fn calibrate_frame(&mut self) -> Result<()> {
        let [tip, mid, moving] = self.jaw_points_local()?;
        // Approach axis: from the fixed-jaw middle pad towards its tip, snapped to
        // the dominant site axis (the site frame is axis-aligned with the jaw).
        let a = [tip[0] - mid[0], tip[1] - mid[1], tip[2] - mid[2]];
        let k = (0..3)
            .max_by(|&i, &j| a[i].abs().total_cmp(&a[j].abs()))
            .unwrap();
        let mut snapped = [0.; 3];
        snapped[k] = a[k].signum();
        self.approach_axis_local = snapped;
        self.jaw_local = [tip, mid, moving];
        Ok(())
    }

    /// Live fixed-jaw tip, fixed-jaw middle pad and moving-jaw tip, expressed in
    /// the gripperframe site frame. Diagnostic for grasp-centre calibration.
    pub fn jaw_points_local(&self) -> Result<[[f64; 3]; 3]> {
        let m = &self.sim.model;
        let geom = |name: &str| {
            m.object_id::<obj::Geom>(name)
                .ok_or_else(|| Error::Model(format!("missing jaw geom {name}")))
        };
        let ids = [
            geom("fixed_jaw_sph_tip1")?,
            geom("fixed_jaw_box6")?,
            geom("moving_jaw_sph_tip1")?,
        ];
        let d = &self.sim.data;
        // SAFETY: data belongs to this model.
        let gx = unsafe { d.geom_xpos(m) };
        let sx = unsafe { d.site_xpos(m) };
        let sm = unsafe { d.site_xmat(m) };
        let s = self.grip_site.index();
        let r = &sm[9 * s..9 * s + 9];
        Ok(ids.map(|id| {
            let i = id.index();
            let w = [
                gx[3 * i] - sx[3 * s],
                gx[3 * i + 1] - sx[3 * s + 1],
                gx[3 * i + 2] - sx[3 * s + 2],
            ];
            std::array::from_fn(|c| (0..3).map(|k| r[3 * k + c] * w[k]).sum())
        }))
    }

    // ---- state access -------------------------------------------------
    pub fn cube_pose(&self) -> ([f64; 3], [f64; 4]) {
        let q = self.sim.data.qpos(self.cube_joint, &self.sim.model);
        ([q[0], q[1], q[2]], [q[3], q[4], q[5], q[6]])
    }
    pub fn cube_velocity(&self) -> [f64; 6] {
        self.sim.data.qvel(self.cube_joint, &self.sim.model)
    }
    pub fn target(&self) -> [f64; 3] {
        self.target
    }
    pub fn set_target(&mut self, target: [f64; 3]) -> Result<()> {
        if !target.iter().all(|v| v.is_finite()) {
            return Err(Error::Input("non-finite place target".into()));
        }
        self.target = target;
        self.sim
            .data
            .set_mocap_pos(
                self.target_body,
                [target[0], target[1], 0.0],
                &self.sim.model,
            )
            .ok_or_else(|| Error::Model("place_target is not mocap".into()))?;
        mj::mj_forward(&self.sim.model, &mut self.sim.data);
        Ok(())
    }
    /// Place the cube at rest (sets free-joint qpos, zero velocity). Reset-time only.
    pub fn set_cube(&mut self, xy: [f64; 2], yaw: f64) -> Result<()> {
        if !xy.iter().chain([&yaw]).all(|v| v.is_finite()) {
            return Err(Error::Input("non-finite cube pose".into()));
        }
        let (s, c) = (yaw / 2.).sin_cos();
        self.sim.data.set_qpos(
            self.cube_joint,
            [xy[0], xy[1], self.config.cube_half, c, 0., 0., s],
            &self.sim.model,
        );
        self.sim
            .data
            .set_qvel(self.cube_joint, [0.; 6], &self.sim.model);
        mj::mj_forward(&self.sim.model, &mut self.sim.data);
        Ok(())
    }

    /// Gripper pinch centre (world) and approach axis from live data.
    pub fn gripper_pose(&self) -> ([f64; 3], [f64; 3]) {
        // SAFETY: matching model/data.
        let sx = unsafe { self.sim.data.site_xpos(&self.sim.model) };
        let sm = unsafe { self.sim.data.site_xmat(&self.sim.model) };
        let s = self.grip_site.index();
        self.frame_pose(
            [sx[3 * s], sx[3 * s + 1], sx[3 * s + 2]],
            &sm[9 * s..9 * s + 9],
        )
    }
    fn frame_pose(&self, p: [f64; 3], r: &[f64]) -> ([f64; 3], [f64; 3]) {
        let o = mat_vec(r, self.grasp_offset);
        (
            [p[0] + o[0], p[1] + o[1], p[2] + o[2]],
            mat_vec(r, self.approach_axis_local),
        )
    }
    /// Side-effect-free FK of the pinch centre and approach axis.
    pub fn gripper_fk(&self, q: &[f64]) -> Result<([f64; 3], [f64; 3])> {
        self.sim.validate_command(&JointCommand {
            position: q.to_vec(),
        })?;
        let m = &self.sim.model;
        let mut d = self.sim.fk_data.borrow_mut();
        mj::mj_resetData(m, &mut d);
        for (mapping, v) in self.sim.joint_ids().into_iter().zip(q) {
            d.set_qpos(mapping, [*v], m);
        }
        mj::mj_kinematics(m, &mut d);
        // SAFETY: matching model/data.
        let sx = unsafe { d.site_xpos(m) };
        let sm = unsafe { d.site_xmat(m) };
        let s = self.grip_site.index();
        Ok(self.frame_pose(
            [sx[3 * s], sx[3 * s + 1], sx[3 * s + 2]],
            &sm[9 * s..9 * s + 9],
        ))
    }

    pub fn contacts(&self) -> Contacts {
        let m = &self.sim.model;
        let d = &self.sim.data;
        let mut out = Contacts::default();
        for c in d.contact() {
            let (Some(a), Some(b)) = c.geom() else {
                continue;
            };
            let other = if a == self.cube_geom {
                b
            } else if b == self.cube_geom {
                a
            } else {
                continue;
            };
            if c.dist() > 1e-4 {
                continue; // within margin only, not touching
            }
            if other == self.table_geom {
                out.table = true;
                continue;
            }
            let body = m.geom_bodyid(other);
            if body == self.fixed_jaw_body {
                out.fixed_jaw = true;
            } else if body == self.moving_jaw_body {
                out.moving_jaw = true;
            } else if body.index() != 0 {
                out.other_robot = true;
            }
        }
        out
    }

    pub fn state(&self) -> PickPlaceState {
        let js = self.sim.state();
        let (gp, ga) = self.gripper_pose();
        let (cp, cq) = self.cube_pose();
        let v = self.cube_velocity();
        PickPlaceState {
            joint_position: js.position,
            joint_velocity: js.velocity,
            gripper_position: gp,
            gripper_axis: ga,
            cube_position: cp,
            cube_quat: cq,
            cube_linear_velocity: [v[0], v[1], v[2]],
            target_position: self.target,
            contacts: self.contacts(),
            time: self.sim.time(),
        }
    }

    // ---- episode ------------------------------------------------------
    /// Reset physics, then sample cube and goal positions (reproducible by seed).
    pub fn reset(&mut self, seed: u64) -> Result<()> {
        self.sim.reset_seed(seed)?;
        let mut rng = seed ^ 0x5eed_9a1c_e000_0000;
        let c = self.config.clone();
        for _ in 0..1000 {
            let sample = |rng: &mut u64| {
                let r = lerp(c.radius_range, uniform(rng));
                let a = lerp(c.azimuth_range, uniform(rng));
                [r * a.cos(), r * a.sin()]
            };
            let cube = sample(&mut rng);
            let goal = sample(&mut rng);
            let yaw = lerp(c.yaw_range, uniform(&mut rng));
            if (cube[0] - goal[0]).hypot(cube[1] - goal[1]) >= c.min_separation {
                self.set_cube(cube, yaw)?;
                self.set_target([goal[0], goal[1], c.cube_half])?;
                return Ok(());
            }
        }
        Err(Error::Input("could not sample separated cube/goal".into()))
    }
    pub fn step(&mut self, command: &JointCommand, period: f64) -> Result<()> {
        self.sim.step_control(command, period)?;
        let (p, _) = self.cube_pose();
        if !p.iter().all(|v| v.is_finite()) {
            return Err(Error::Unstable);
        }
        Ok(())
    }

    // ---- kinematics ---------------------------------------------------
    /// Closing direction of the jaws (site local z) in world, from live data.
    pub fn closing_direction(&self) -> [f64; 3] {
        // SAFETY: matching model/data.
        let sm = unsafe { self.sim.data.site_xmat(&self.sim.model) };
        let s = self.grip_site.index();
        mat_vec(&sm[9 * s..9 * s + 9], [0., 0., 1.])
    }
    /// Side-effect-free FK of the jaw closing direction (world).
    pub fn fk_closing(&self, q: &[f64]) -> Result<[f64; 3]> {
        Ok(self.fk_full(q)?.2)
    }
    fn fk_full(&self, q: &[f64]) -> Result<([f64; 3], [f64; 3], [f64; 3])> {
        let (p, a) = self.gripper_fk(q)?;
        let d = self.sim.fk_data.borrow();
        // SAFETY: matching model/data.
        let sm = unsafe { d.site_xmat(&self.sim.model) };
        let s = self.grip_site.index();
        Ok((p, a, mat_vec(&sm[9 * s..9 * s + 9], [0., 0., 1.])))
    }

    /// Damped least squares IK for the pinch-centre position, approach axis and
    /// (optionally) the jaw closing direction, iterated on the FK model with no
    /// live-state side effects. Only ARM joints change; the gripper is preserved.
    /// Returns the joint vector and the remaining position error (metres).
    pub fn solve_ik(
        &self,
        q0: &[f64],
        target: &IkTarget,
        iterations: usize,
    ) -> Result<(Vec<f64>, f64)> {
        let bounds = self.sim.action_bounds();
        let mut q: Vec<f64> = q0
            .iter()
            .zip(&bounds)
            .map(|(v, [lo, hi])| v.clamp(*lo, *hi))
            .collect();
        let w = target.axis_weight;
        let residual = |q: &[f64]| -> Result<Vec<f64>> {
            let (p, a, z) = self.fk_full(q)?;
            // Vector difference, not a cross product: the latter also vanishes
            // for the opposite (fingers-up) direction.
            let e = [
                target.axis[0] - a[0],
                target.axis[1] - a[1],
                target.axis[2] - a[2],
            ];
            let mut r = vec![
                target.position[0] - p[0],
                target.position[1] - p[1],
                target.position[2] - p[2],
                w * e[0],
                w * e[1],
                w * e[2],
            ];
            if let Some(c) = target.closing {
                // Parallel either way: the jaw pinch is symmetric under 180 deg.
                let c = if z.iter().zip(c).map(|(a, b)| a * b).sum::<f64>() < 0. {
                    c.map(|v| -v)
                } else {
                    c
                };
                let e = cross(z, c);
                r.extend(e.map(|v| w * v));
            }
            Ok(r)
        };
        let mut err = residual(&q)?;
        let n = err.len();
        for _ in 0..iterations {
            let norm = err[..3].iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm < 1e-4 && err[3..].iter().all(|v| v.abs() < 1e-3 * w) {
                break;
            }
            let mut jac = vec![vec![0.; n]; ARM.len()];
            for (col, &i) in ARM.iter().enumerate() {
                let h = 1e-5;
                let mut plus = q.clone();
                plus[i] = (q[i] + h).min(bounds[i][1]);
                let mut minus = q.clone();
                minus[i] = (q[i] - h).max(bounds[i][0]);
                let (rp, rm) = (residual(&plus)?, residual(&minus)?);
                let dh = plus[i] - minus[i];
                // residual = target - f(q)  =>  J_f = -(d residual)/dq
                for k in 0..n {
                    jac[col][k] = -(rp[k] - rm[k]) / dh;
                }
            }
            let lambda2 = 1e-5;
            let a: Vec<Vec<f64>> = (0..n)
                .map(|r| {
                    (0..n)
                        .map(|c| {
                            jac.iter().map(|j| j[r] * j[c]).sum::<f64>()
                                + if r == c { lambda2 } else { 0. }
                        })
                        .collect()
                })
                .collect();
            let y = solve(a, err.clone());
            for (col, &i) in ARM.iter().enumerate() {
                let dq: f64 = (0..n).map(|k| jac[col][k] * y[k]).sum();
                q[i] = (q[i] + dq.clamp(-0.2, 0.2)).clamp(bounds[i][0], bounds[i][1]);
            }
            err = residual(&q)?;
        }
        let pos_err = err[..3].iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok((q, pos_err))
    }
}

/// IK goal for the pinch centre. `axis_weight` converts axis error (rad) to metres.
#[derive(Debug, Clone)]
pub struct IkTarget {
    pub position: [f64; 3],
    pub axis: [f64; 3],
    pub closing: Option<[f64; 3]>,
    pub axis_weight: f64,
}
impl IkTarget {
    /// Fingers pointing straight down.
    pub fn top_down(position: [f64; 3], closing: Option<[f64; 3]>) -> Self {
        Self {
            position,
            axis: [0., 0., -1.],
            closing,
            axis_weight: 0.1,
        }
    }
}
