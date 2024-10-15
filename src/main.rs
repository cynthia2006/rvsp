use clap::Parser;

use gl::types::GLint;
use sdl2::audio::AudioSpecDesired;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use realfft::RealFftPlanner;

use const_format::formatcp;
use skia_safe::gpu::gl::FramebufferInfo;
use skia_safe::gpu::{self, backend_render_targets, SurfaceOrigin};
use skia_safe::ColorType;

mod callback;
mod cli;
mod defs;
mod render;
mod utils;

const WINDOW_TITLE: &'static str = formatcp!(
    "{} (v{})",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_VERSION")
);

fn find_sdl_gl_driver() -> Option<u32> {
    for (index, item) in sdl2::render::drivers().enumerate() {
        if item.name == "opengl" {
            return Some(index as u32);
        }
    }
    None
}

fn main() {
    let args = cli::Args::parse();

    let sdl = sdl2::init().unwrap();
    let sdl_audio = sdl.audio().unwrap();
    let sdl_video = sdl.video().unwrap();
    let mut sdl_events = sdl.event_pump().unwrap();

    let mut planner = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(defs::FFT_SIZE);
    let mut fft_scratch = fft.make_scratch_vec();
    let blackman_window: Vec<f32> = utils::blackman_window(defs::FFT_SIZE).collect();
    let mut windowed_signal = fft.make_input_vec();
    let mut frequency_bins = fft.make_output_vec();

    let mut device = sdl_audio
        .open_capture(
            None,
            &AudioSpecDesired {
                channels: Some(1),
                freq: Some(48000),
                samples: Some(2048),
            },
            |_| callback::Callback {
                buffer: fft.make_input_vec(),
            },
        )
        .unwrap();

    let window = sdl_video
        .window(WINDOW_TITLE, args.width, args.height)
        .position_centered()
        // .resizable()
        .build()
        .unwrap();

    let mut canvas = window
        .into_canvas()
        .index(find_sdl_gl_driver().unwrap())
        .present_vsync()
        .build()
        .unwrap();

    gl::load_with(|name| sdl_video.gl_get_proc_address(name) as *const _);

    let interface = skia_safe::gpu::gl::Interface::new_load_with(|name| {
        sdl_video.gl_get_proc_address(name) as *const _
    })
    .unwrap();

    canvas.window().gl_set_context_to_current().unwrap();

    let mut gr_context = skia_safe::gpu::direct_contexts::make_gl(interface, None).unwrap();
    let fb_info = {
        let mut fboid: GLint = 0;
        unsafe { gl::GetIntegerv(gl::FRAMEBUFFER_BINDING, &mut fboid) };

        FramebufferInfo {
            fboid: fboid.try_into().unwrap(),
            format: skia_safe::gpu::gl::Format::RGBA8.into(),
            ..Default::default()
        }
    };
    let backend_render_target = backend_render_targets::make_gl(
        (args.width as i32, args.height as i32),
        None,
        {
            let mut stencil_size: GLint = 0;

            unsafe {
                gl::GetFramebufferAttachmentParameteriv(
                    gl::DRAW_FRAMEBUFFER,
                    gl::STENCIL,
                    gl::FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE,
                    &mut stencil_size,
                )
            };

            stencil_size as usize
        },
        fb_info,
    );

    let skia_surface = gpu::surfaces::wrap_backend_render_target(
        &mut gr_context,
        &backend_render_target,
        SurfaceOrigin::BottomLeft,
        ColorType::N32,
        None,
        None,
    )
    .unwrap();

    device.resume();

    let mut renderer = render::Renderer::new(
        skia_surface,
        defs::FFT_SIZE,
        args.tau,
        args.min_frequency,
        args.max_frequency,
        args.db_min,
        args.db_max,
        args.fg,
        args.bg,
        args.width as f32,
        args.height as f32,
    );

    'running: loop {
        for event in sdl_events.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                // Event::Window { win_event, .. } => match win_event {
                //     WindowEvent::Resized(w, h) => {
                //         renderer.set_width(w as f32);
                //         renderer.set_height(h as f32);
                //     },
                //     _ => {}
                // },
                _ => {}
            }
        }

        let callback_context = device.lock();

        for (i, s) in callback_context.buffer.iter().enumerate() {
            windowed_signal[i] = blackman_window[i] * (*s);
        }

        // An explicit drop is required because if the lock is held for too long, callback will be inhibited to recieve
        // data on time, ultimately causing horrendous lags.
        drop(callback_context);

        fft.process_with_scratch(&mut windowed_signal, &mut frequency_bins, &mut fft_scratch)
            .unwrap();

        renderer.render(&frequency_bins);

        gr_context.flush_and_submit();

        canvas.present();
    }
}
