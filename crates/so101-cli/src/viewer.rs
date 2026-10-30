use glfw::{Action, Context, Key, MouseButton, WindowEvent};
use mujoco_runtime::Simulator;
use robotics_core::controller::Controller;
use rusty_mujoco::{self as mj, mjrContext, mjrRect, mjtFontScale, mjvCamera, mjvOption, mjvScene};
use std::time::Instant;

/// Must be called on the main thread on macOS. All simulation steps use core.
pub fn run(
    sim: &mut Simulator,
    controller: &mut dyn Controller,
    steps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut glfw = glfw::init(|error, description| eprintln!("GLFW {error:?}: {description}"))?;
    glfw.window_hint(glfw::WindowHint::ContextVersion(2, 1));
    let (mut window, events) = glfw
        .create_window(1000, 800, "Robot / MuJoCo", glfw::WindowMode::Windowed)
        .ok_or("could not create GLFW window")?;
    window.make_current();
    glfw.set_swap_interval(glfw::SwapInterval::Sync(1));
    window.set_all_polling(true);

    // These resources drop while the window's OpenGL context is still current.
    let is_so101 = sim.spec().name == "so101";
    let (context, mut scene, mut camera) = sim.with_mujoco(|model, _| {
        let mut camera = mjvCamera::default_free(model);
        if is_so101 {
            camera.set_lookat([0.0, 0.0, 0.2]);
            camera.set_distance(0.9);
        }
        camera.set_azimuth(135.0);
        camera.set_elevation(-25.0);
        (
            mjrContext::new(model, mjtFontScale::X150),
            mjvScene::new(model, 2000),
            camera,
        )
    });
    let options = mjvOption::default();
    let mut previous = Instant::now();
    let mut accumulator = 0.0;
    let mut count = 0;
    let mut frames = 0;
    let mut paused = false;
    let mut cursor = window.get_cursor_pos();
    eprintln!("Viewer: drag left to orbit; scroll to zoom; Space pause; R reset; Esc close.");

    while !window.should_close() && count < steps {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
                WindowEvent::Key(Key::Space, _, Action::Press, _) => paused = !paused,
                WindowEvent::Key(Key::R, _, Action::Press, _) => {
                    sim.reset()?;
                    accumulator = 0.0;
                    count = 0;
                }
                WindowEvent::Scroll(_, y) => {
                    camera.set_distance((camera.distance() * (-0.1 * y).exp()).clamp(0.1, 5.0));
                }
                WindowEvent::CursorPos(x, y) => {
                    if window.get_mouse_button(MouseButton::Button1) == Action::Press {
                        camera.set_azimuth(camera.azimuth() - (x - cursor.0) * 0.3);
                        camera.set_elevation(
                            (camera.elevation() - (y - cursor.1) * 0.3).clamp(-89.0, 89.0),
                        );
                    }
                    cursor = (x, y);
                }
                _ => {}
            }
        }
        let now = Instant::now();
        // Limit catch-up after window stalls without changing physics timestep.
        let elapsed = now.duration_since(previous).as_secs_f64().min(0.1);
        previous = now;
        if !paused {
            accumulator += elapsed;
        }
        while accumulator >= sim.timestep() && count < steps {
            let command = controller.command(&sim.state(), sim.time());
            sim.step(&command)?;
            accumulator -= sim.timestep();
            count += 1;
        }
        let (width, height) = window.get_framebuffer_size();
        if width > 0 && height > 0 {
            sim.update_scene(&options, &mut camera, &mut scene);
            mj::mjr_render(
                mjrRect::new(0, 0, width as u32, height as u32),
                &mut scene,
                &context,
            );
            window.swap_buffers();
            frames += 1;
        }
    }
    eprintln!("Viewer rendered {frames} frames, advanced {count} steps.");
    Ok(())
}
