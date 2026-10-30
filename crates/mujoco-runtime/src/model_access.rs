use rusty_mujoco::{self as mj, ObjectId, obj};

#[cfg(feature = "render")]
pub(crate) fn render_quality(model: &mut mj::mjModel) {
    unsafe extern "C" {
        fn robot_sim_render_quality(model: *mut mj::bindgen::mjModel);
    }
    // SAFETY: exclusive access to the renderer's owned model. Only visual
    // quality fields are written; no allocation/topology/physics is changed.
    unsafe { robot_sim_render_quality(&mut **model) };
}

unsafe extern "C" {
    fn robot_sim_actuator_joint(model: *const mj::bindgen::mjModel, actuator: i32) -> i32;
}

pub fn actuator_joint(model: &mj::mjModel, actuator: ObjectId<obj::Actuator>) -> Option<usize> {
    // SAFETY: wrapper deref borrows its live model allocation; the C function is
    // read-only, checks actuator bounds/type, and stores no pointer.
    let index = unsafe { robot_sim_actuator_joint(&**model, actuator.index() as i32) };
    usize::try_from(index).ok()
}

unsafe extern "C" {
    fn robot_sim_read_parameters(
        model: *const mj::bindgen::mjModel,
        actuator: i32,
        out: *mut f64,
    ) -> i32;
    fn robot_sim_write_parameters(
        model: *mut mj::bindgen::mjModel,
        actuator: i32,
        values: *const f64,
    ) -> i32;
}
pub(crate) fn parameters(
    model: &mj::mjModel,
    actuator: ObjectId<obj::Actuator>,
) -> robotics_core::Result<[f64; 7]> {
    let mut out = [0.; 7];
    // SAFETY: live model, seven-element output buffer, checked actuator/hinge/DOF in C; no retained pointers.
    let ok =
        unsafe { robot_sim_read_parameters(&**model, actuator.index() as i32, out.as_mut_ptr()) };
    if ok == 0 {
        return Err(robotics_core::Error::Model(
            "invalid parameter mapping".into(),
        ));
    }
    Ok(out)
}
pub(crate) fn set_parameters(
    model: &mut mj::mjModel,
    actuator: ObjectId<obj::Actuator>,
    values: &[f64; 7],
) -> robotics_core::Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(robotics_core::Error::Input(
            "nonfinite physical parameter".into(),
        ));
    }
    // SAFETY: exclusive model access at reset; seven valid values; C bounds/type checks;
    // fields do not change sizes/topology or retain references. mjData is reset afterwards.
    let ok = unsafe {
        robot_sim_write_parameters(&mut **model, actuator.index() as i32, values.as_ptr())
    };
    if ok == 0 {
        return Err(robotics_core::Error::Model(
            "invalid parameter mapping".into(),
        ));
    }
    Ok(())
}
