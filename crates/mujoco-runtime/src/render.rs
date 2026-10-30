//! Offscreen visualization of an owned copy. Never steps or mutates the source.
//! CGL (macOS) or hidden GLFW (other platforms); keep resources on one thread.
use crate::{Error, Result, Simulator};
use rusty_mujoco::{self as mj, bindgen::mjtState};
use std::io::Write;

/// Self-contained compiled model, including meshes and textures (MJB).
pub struct ModelSnapshot(Vec<u8>);

pub fn model_snapshot(sim: &Simulator) -> ModelSnapshot {
    let mut bytes = vec![0; mj::mj_sizeModel(&sim.model)];
    mj::mj_saveModel(&sim.model, None::<String>, Some(&mut bytes));
    ModelSnapshot(bytes)
}

/// Includes every qpos and the mocap target, with no Python-side joint mapping.
pub fn state_snapshot(sim: &Simulator) -> Vec<f64> {
    let mut state = vec![0.; mj::mj_stateSize(&sim.model, mjtState::INTEGRATION)];
    mj::mj_getState(&sim.model, &sim.data, &mut state, mjtState::INTEGRATION);
    state
}

pub struct OffscreenRenderer {
    // Declaration order matters: GL resources before window, data before model.
    context: mj::mjrContext,
    scene: mj::mjvScene,
    data: mj::mjData,
    model: mj::mjModel,
    camera: mj::mjvCamera,
    options: mj::mjvOption,
    gl: gl_context::GlContext,
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    depth: Vec<f32>,
}

impl OffscreenRenderer {
    pub fn new(compiled_model: &ModelSnapshot, width: u32, height: u32) -> Result<Self> {
        if !(64..=3840).contains(&width) || !(64..=2160).contains(&height) {
            return Err(Error::Input(
                "render size must be within 64..3840 x 64..2160".into(),
            ));
        }
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn pthread_main_np() -> std::ffi::c_int;
            }
            // SAFETY: pthread_main_np has no arguments or side effects.
            if unsafe { pthread_main_np() } == 0 {
                return Err(Error::Input(
                    "create/use the renderer on the macOS main thread".into(),
                ));
            }
        }
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(&compiled_model.0)?;
        let mut model = mj::mj_loadModel(file.path().to_string_lossy().into_owned());
        // Rendering-only copy: keep software rendering practical, including
        // when no GPU is available. Physics parameters are never changed.
        crate::model_access::render_quality(&mut model);
        let data = mj::mj_makeData(&model);
        let mut gl = gl_context::GlContext::new()?;
        gl.make_current()?;
        let mut context = mj::mjrContext::new(&model, mj::mjtFontScale::X100);
        mj::mjr_resizeOffscreen(width as i32, height as i32, &mut context);
        mj::mjr_setBuffer(mj::mjtFramebuffer::OFFSCREEN, &mut context);
        let scene = mj::mjvScene::new(&model, 2000);
        let mut camera = mj::mjvCamera::default_free(&model);
        camera.set_lookat([0.12, 0., 0.12]);
        camera.set_distance(0.75);
        camera.set_azimuth(135.);
        camera.set_elevation(-35.);
        Ok(Self {
            context,
            scene,
            data,
            model,
            camera,
            options: mj::mjvOption::default(),
            gl,
            width,
            height,
            rgb: vec![0; (width * height * 3) as usize],
            depth: vec![0.; (width * height) as usize],
        })
    }

    /// Top-to-bottom RGB8 pixels. Forward kinematics runs only on the render copy.
    pub fn render(&mut self, state: &[f64]) -> Result<Vec<u8>> {
        if state.len() != mj::mj_stateSize(&self.model, mjtState::INTEGRATION) {
            return Err(Error::Input(
                "render state does not match the compiled model".into(),
            ));
        }
        self.gl.make_current()?;
        mj::mj_setState(&self.model, &mut self.data, state, mjtState::INTEGRATION);
        mj::mj_forward(&self.model, &mut self.data);
        mj::mjv_updateScene(
            &self.model,
            &mut self.data,
            &self.options,
            None,
            &mut self.camera,
            mj::mjtCatBit::ALL,
            &mut self.scene,
        );
        let viewport = mj::mjrRect::new(0, 0, self.width, self.height);
        mj::mjr_render(viewport, &mut self.scene, &self.context);
        mj::mjr_readPixels(&mut self.rgb, &mut self.depth, viewport, &self.context);
        let stride = (self.width * 3) as usize;
        let mut rgb = vec![0; self.rgb.len()];
        for (dst, src) in rgb
            .chunks_exact_mut(stride)
            .zip(self.rgb.chunks_exact(stride).rev())
        {
            dst.copy_from_slice(src);
        }
        Ok(rgb)
    }
}

impl Drop for OffscreenRenderer {
    fn drop(&mut self) {
        // Another renderer could have made its own context current meanwhile.
        let _ = self.gl.make_current();
    }
}

#[cfg(target_os = "macos")]
mod gl_context {
    use super::*;
    use std::ffi::{c_int, c_void};

    #[link(name = "OpenGL", kind = "framework")]
    unsafe extern "C" {
        fn CGLChoosePixelFormat(
            attributes: *const c_int,
            format: *mut *mut c_void,
            count: *mut c_int,
        ) -> c_int;
        fn CGLDestroyPixelFormat(format: *mut c_void) -> c_int;
        fn CGLCreateContext(
            format: *mut c_void,
            share: *mut c_void,
            context: *mut *mut c_void,
        ) -> c_int;
        fn CGLDestroyContext(context: *mut c_void) -> c_int;
        fn CGLSetCurrentContext(context: *mut c_void) -> c_int;
    }

    pub struct GlContext(*mut c_void);
    impl GlContext {
        pub fn new() -> Result<Self> {
            // Official CGLTypes.h / CGLRenderers.h. Prefer acceleration, then
            // Apple's software renderer when a GPU is unavailable (e.g. CI).
            // Both allow offline renderers and need no Cocoa/input services.
            let accelerated = [73, 96, 8, 24, 12, 24, 0];
            let software = [70, 0x00020400, 96, 8, 24, 12, 24, 0];
            let mut format = std::ptr::null_mut();
            let mut count = 0;
            let mut context = std::ptr::null_mut();
            // SAFETY: terminated attribute array; valid out pointers. The pixel
            // format is released on every successful allocation; context is owned.
            unsafe {
                let mut code = CGLChoosePixelFormat(accelerated.as_ptr(), &mut format, &mut count);
                if code != 0 || format.is_null() {
                    code = CGLChoosePixelFormat(software.as_ptr(), &mut format, &mut count);
                }
                if code != 0 || format.is_null() {
                    return Err(Error::Model(format!("CGLChoosePixelFormat failed: {code}")));
                }
                let code = CGLCreateContext(format, std::ptr::null_mut(), &mut context);
                CGLDestroyPixelFormat(format);
                if code != 0 || context.is_null() {
                    return Err(Error::Model(format!("CGLCreateContext failed: {code}")));
                }
            }
            Ok(Self(context))
        }
        pub fn make_current(&mut self) -> Result<()> {
            // SAFETY: self owns this live CGL context on its creating thread.
            let code = unsafe { CGLSetCurrentContext(self.0) };
            if code != 0 {
                return Err(Error::Model(format!("CGLSetCurrentContext failed: {code}")));
            }
            Ok(())
        }
    }
    impl Drop for GlContext {
        fn drop(&mut self) {
            // SAFETY: GL resources were dropped before this owned context.
            unsafe {
                CGLSetCurrentContext(std::ptr::null_mut());
                CGLDestroyContext(self.0);
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod gl_context {
    use super::*;
    use glfw::Context;

    pub struct GlContext {
        window: glfw::PWindow,
        _glfw: glfw::Glfw,
    }
    impl GlContext {
        pub fn new() -> Result<Self> {
            let mut glfw =
                glfw::init(|error, description| eprintln!("GLFW {error:?}: {description}"))
                    .map_err(|e| Error::Model(format!("GLFW initialization: {e}")))?;
            glfw.window_hint(glfw::WindowHint::Visible(false));
            glfw.window_hint(glfw::WindowHint::ContextVersion(2, 1));
            let (window, _) = glfw
                .create_window(64, 64, "Pick & Place offscreen", glfw::WindowMode::Windowed)
                .ok_or_else(|| Error::Model("could not create hidden GLFW context".into()))?;
            Ok(Self {
                window,
                _glfw: glfw,
            })
        }
        pub fn make_current(&mut self) -> Result<()> {
            self.window.make_current();
            Ok(())
        }
    }
}
