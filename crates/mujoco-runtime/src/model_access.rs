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

unsafe extern "C" {
    fn robot_sim_read_object(
        model: *const mj::bindgen::mjModel,
        body: i32,
        geom: i32,
        out: *mut f64,
    ) -> i32;
    fn robot_sim_write_object(
        model: *mut mj::bindgen::mjModel,
        data: *mut mj::bindgen::mjData,
        body: i32,
        geom: i32,
        values: *const f64,
    ) -> i32;
    fn robot_sim_max_geom_priority(model: *const mj::bindgen::mjModel) -> i32;
}
/// mass, principal inertia[3], sliding friction, priority of a body's geom.
pub(crate) fn object_parameters(
    model: &mj::mjModel,
    body: ObjectId<obj::Body>,
    geom: ObjectId<obj::Geom>,
) -> robotics_core::Result<[f64; 6]> {
    let mut out = [0.; 6];
    // SAFETY: live model, six-element buffer, C checks body/geom ownership; no retained pointers.
    let ok = unsafe {
        robot_sim_read_object(
            &**model,
            body.index() as i32,
            geom.index() as i32,
            out.as_mut_ptr(),
        )
    };
    if ok == 0 {
        return Err(robotics_core::Error::Model("invalid object mapping".into()));
    }
    Ok(out)
}
pub(crate) fn set_object_parameters(
    model: &mut mj::mjModel,
    data: &mut mj::mjData,
    body: ObjectId<obj::Body>,
    geom: ObjectId<obj::Geom>,
    values: &[f64; 6],
) -> robotics_core::Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(robotics_core::Error::Input(
            "nonfinite object parameter".into(),
        ));
    }
    // SAFETY: exclusive reset-time access to matching model/data; no size or
    // topology change; mjData is reset by the caller afterwards.
    let ok = unsafe {
        robot_sim_write_object(
            &mut **model,
            &mut **data,
            body.index() as i32,
            geom.index() as i32,
            values.as_ptr(),
        )
    };
    if ok == 0 {
        return Err(robotics_core::Error::Model("invalid object mapping".into()));
    }
    Ok(())
}
pub(crate) fn max_geom_priority(model: &mj::mjModel) -> i32 {
    // SAFETY: read-only scan of a live model.
    unsafe { robot_sim_max_geom_priority(&**model) }
}

unsafe extern "C" {
    fn robot_sim_read_geom_contact(
        model: *const mj::bindgen::mjModel,
        geom: i32,
        out: *mut f64,
    ) -> i32;
    fn robot_sim_write_geom_contact(
        model: *mut mj::bindgen::mjModel,
        geom: i32,
        values: *const f64,
    ) -> i32;
    fn robot_sim_geom_name(
        model: *const mj::bindgen::mjModel,
        geom: i32,
    ) -> *const std::ffi::c_char;
    fn robot_sim_read_option(model: *const mj::bindgen::mjModel, out: *mut f64);
    fn robot_sim_geom_contacts(
        model: *const mj::bindgen::mjModel,
        data: *const mj::bindgen::mjData,
        geom: i32,
        out: *mut f64,
        max: i32,
    ) -> i32;
}
/// friction[3], condim, solref[2], priority of one geom.
pub(crate) fn geom_contact(
    model: &mj::mjModel,
    geom: ObjectId<obj::Geom>,
) -> robotics_core::Result<[f64; 7]> {
    let mut out = [0.; 7];
    // SAFETY: live model, seven-element buffer, C bounds check; read-only.
    let ok =
        unsafe { robot_sim_read_geom_contact(&**model, geom.index() as i32, out.as_mut_ptr()) };
    if ok == 0 {
        return Err(robotics_core::Error::Model("invalid geom".into()));
    }
    Ok(out)
}
pub(crate) fn set_geom_contact(
    model: &mut mj::mjModel,
    geom: ObjectId<obj::Geom>,
    values: &[f64; 7],
) -> robotics_core::Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(robotics_core::Error::Input(
            "nonfinite contact parameter".into(),
        ));
    }
    // SAFETY: exclusive model access; only per-geom contact values change.
    let ok =
        unsafe { robot_sim_write_geom_contact(&mut **model, geom.index() as i32, values.as_ptr()) };
    if ok == 0 {
        return Err(robotics_core::Error::Input(
            "invalid geom contact parameters".into(),
        ));
    }
    Ok(())
}
/// Geom name, or "" for an unnamed geom.
pub(crate) fn geom_name(model: &mj::mjModel, geom: usize) -> String {
    // SAFETY: the C helper bounds-checks and never returns NULL; the string is
    // owned by the model and copied before the borrow ends.
    unsafe { std::ffi::CStr::from_ptr(robot_sim_geom_name(&**model, geom as i32)) }
        .to_string_lossy()
        .into_owned()
}
/// timestep, integrator, solver, iterations, ls_iterations, cone, impratio,
/// noslip_iterations, tolerance, override flag.
pub(crate) fn solver_option(model: &mj::mjModel) -> [f64; 10] {
    let mut out = [0.; 10];
    // SAFETY: read-only, ten-element buffer.
    unsafe { robot_sim_read_option(&**model, out.as_mut_ptr()) };
    out
}
pub(crate) const CONTACT_FIELDS: usize = 27;
/// Raw contact rows of one geom (see model_access.c for the layout).
pub(crate) fn geom_contacts(
    model: &mj::mjModel,
    data: &mj::mjData,
    geom: ObjectId<obj::Geom>,
) -> Vec<[f64; CONTACT_FIELDS]> {
    const MAX: usize = 64;
    let mut out = vec![0.; MAX * CONTACT_FIELDS];
    // SAFETY: matching model/data, buffer of MAX rows; read-only.
    let n = unsafe {
        robot_sim_geom_contacts(
            &**model,
            &**data,
            geom.index() as i32,
            out.as_mut_ptr(),
            MAX as i32,
        )
    };
    let (rows, _) = out.as_chunks::<CONTACT_FIELDS>();
    rows.iter().take(n.max(0) as usize).copied().collect()
}
