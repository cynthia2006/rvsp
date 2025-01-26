use std::f32::consts::PI;

use gl::types::GLint;

use sdl2::audio::{AudioCallback, AudioSpecDesired};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;

use realfft::num_complex::ComplexFloat;
use realfft::RealFftPlanner;

use skia_safe::gpu::gl::FramebufferInfo;
use skia_safe::gpu::{self, backend_render_targets, SurfaceOrigin};
use skia_safe::{Color, ColorType, Paint, PaintStyle, Path, Surface};


const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FG: Color = Color::BLACK;
const BG: Color = Color::YELLOW;

const SAMPLERATE: i32 = 48000;
const FFT_SIZE: u16 = 2048;
const MIN_FREQ: i32 = 50;
const MAX_FREQ: i32 = 10000; 
const NORMALIZATION: f32 = 2.0 / FFT_SIZE as f32;
const SMOOTHING_TIME_CONST: f32 = 0.6;


fn get_window_title(gain: f32) -> String {
    format!(
        "{} v{} ({:.2}dB)",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        gain
    )
}

fn find_sdl_gl_driver() -> Option<u32> {
    for (index, item) in sdl2::render::drivers().enumerate() {
        if item.name == "opengl" {
            return Some(index as u32);
        }
    }
    None
}

// https://webaudio.github.io/web-audio-api/#blackman-window
fn blackman_window(n: usize) -> impl Iterator<Item = f32> {
    const A0: f32 = 0.42f32;
    const A1: f32 = 0.5f32;
    const A2: f32 = 0.08f32;

    (0..n).map(move |i| {
        let phi: f32 = 2.0 * PI * i as f32 / n as f32;

        A0 - A1 * phi.cos() + A2 * (2.0 * phi).cos()
    })
}

struct Callback(pub Vec<f32>);

impl AudioCallback for Callback {
    type Channel = f32;

    fn callback(&mut self, samples: &mut [Self::Channel]) {
        if samples.len() > self.0.len() {
            self.0.clear();
            self.0
                .extend(samples.iter().nth(samples.len() - self.0.len()));
        } else {
            self.0.drain(0..samples.len());
            self.0.extend(samples.iter());
        }
    }
}

fn main() {
    let sdl = sdl2::init().unwrap();
    let sdl_audio = sdl.audio().unwrap();
    let sdl_video = sdl.video().unwrap();
    let mut sdl_events = sdl.event_pump().unwrap();

    let mut planner = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE as usize);
    let mut fft_scratch = fft.make_scratch_vec();
    let blackman_window: Vec<f32> = blackman_window(FFT_SIZE as usize).collect();
    let mut windowed_signal = fft.make_input_vec();
    let mut frequency_bins = fft.make_output_vec();
    let mut gain = 9.0;

    const BIN_WIDTH: f32 = FFT_SIZE as f32 / SAMPLERATE as f32;
    const LOW_BIN: usize = (BIN_WIDTH * MIN_FREQ as f32) as usize;
    const HIGH_BIN: usize = (BIN_WIDTH * MAX_FREQ as f32) as usize;

    let mut smoothed_fft = vec![0.0; HIGH_BIN - LOW_BIN];
    
    let mut plot = Path::new();
    let mut plot_paint = Paint::default();
    
    plot_paint
        .set_color(FG)
        .set_stroke_width(1.2)
        .set_style(PaintStyle::Stroke)
        .set_anti_alias(true);


    let mut device = sdl_audio
        .open_capture(
            None,
            &AudioSpecDesired {
                channels: Some(1),
                freq: Some(SAMPLERATE),
                samples: Some(FFT_SIZE / 2),
            },
            |_| Callback(fft.make_input_vec()),
        )
        .unwrap();

    let window = sdl_video
        .window(&get_window_title(gain), WIDTH, HEIGHT)
        .position_centered()
        .resizable()
        .build()
        .unwrap();

    let mut canvas = window
        .into_canvas()
        .index(find_sdl_gl_driver().unwrap())
        .present_vsync()
        .build()
        .unwrap();

    let load_gl_proc = |name: &str| {
        sdl_video.gl_get_proc_address(name.as_ref()) as *const _
    };

    gl::load_with(load_gl_proc);

    let interface = skia_safe::gpu::gl::Interface::new_load_with(load_gl_proc).unwrap();

    canvas.window().gl_set_context_to_current().unwrap();

    let mut gr_context = skia_safe::gpu::direct_contexts::make_gl(interface, None).unwrap();
    let fb_info = {
        let mut fboid: GLint = 0;
        unsafe {
            gl::GetIntegerv(gl::FRAMEBUFFER_BINDING, &mut fboid);
        }

        FramebufferInfo {
            fboid: fboid.try_into().unwrap(),
            format: skia_safe::gpu::gl::Format::RGBA8.into(),
            ..Default::default()
        }
    };

    let mut stencil_size: GLint = 0;

    unsafe {
        gl::GetFramebufferAttachmentParameteriv(
            gl::DRAW_FRAMEBUFFER,
            gl::STENCIL,
            gl::FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE,
            &mut stencil_size,
        )
    };

    fn create_surface(
        width: i32,
        height: i32,
        fb_info: FramebufferInfo,
        gr_context: &mut skia_safe::gpu::DirectContext,
        stencil_size: i32,
    ) -> Option<Surface> {
        let backend_render_target =
            backend_render_targets::make_gl((width, height), None, stencil_size as usize, fb_info);

        gpu::surfaces::wrap_backend_render_target(
            gr_context,
            &backend_render_target,
            SurfaceOrigin::BottomLeft,
            ColorType::N32,
            None,
            None,
        )
    }

    let mut skia_surface = create_surface(
        WIDTH as i32,
        HEIGHT as i32,
        fb_info,
        &mut gr_context,
        stencil_size,
    )
    .unwrap();
    
    device.resume();

    'running: loop {
        /* Event Loop Begin */
        for event in sdl_events.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::Resized(w, h) => {
                        skia_surface = create_surface(
                            w as i32,
                            h as i32,
                            fb_info,
                            &mut gr_context,
                            stencil_size,
                        )
                        .unwrap();
                    }
                    _ => {}
                },
                Event::KeyDown {
                    keycode: Some(Keycode::J),
                    ..
                } => {
                    gain += 0.1;

                    canvas.window_mut().set_title(&get_window_title(gain)).unwrap();
                }
                Event::KeyDown {
                    keycode: Some(Keycode::K),
                    ..
                } => {
                    gain -= 0.1;

                    if gain < 0.0 {
                        gain = 0.0;
                    }

                    canvas.window_mut().set_title(&get_window_title(gain)).unwrap();
                }
                _ => {}
            }
        }
        /* Event Loop End */

        let callback_context = device.lock();

        for (i, s) in callback_context.0.iter().enumerate() {
            windowed_signal[i] = blackman_window[i] * (*s);
        }

        // An explicit drop is required because if the lock is held for too long, callback will be inhibited to recieve
        // data on time, ultimately causing horrendous lags.
        drop(callback_context);

        fft.process_with_scratch(&mut windowed_signal, &mut frequency_bins, &mut fft_scratch)
           .unwrap();

        /* Drawing calls begin */

        skia_surface.canvas().draw_color(BG, None);

        let (width, height) = (skia_surface.width() as f32, skia_surface.height() as f32);
        let margin = 0.01 * width;

        let x_step = (width - 2.0 * margin) / (HIGH_BIN - LOW_BIN) as f32;
        let mut x = x_step + margin;
        let mut sign = 1.0;

        plot.move_to((margin, 0.5 * height));

        for (i, bin) in frequency_bins[LOW_BIN..HIGH_BIN].iter().enumerate() {
            let y = SMOOTHING_TIME_CONST * smoothed_fft[i]
                + (1.0 - SMOOTHING_TIME_CONST) * gain * bin.abs() * NORMALIZATION;
            
            smoothed_fft[i] = y;
            plot.line_to((x, 0.5 * height * (1.0 - sign * y)));

            x += x_step;
            sign *= -1.0;
        }

        skia_surface.canvas().draw_path(&plot, &plot_paint);
        plot.reset();

        /* Drawing calls end */

        gr_context.flush_and_submit();

        canvas.present();
    }
}
