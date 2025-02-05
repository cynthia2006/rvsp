use std::f32::consts::PI;

use circular_buffer::CircularBuffer;
use gl::types::GLint;

use sdl2::video::GLProfile;
use sdl2::audio::{AudioCallback, AudioSpecDesired};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;

use realfft::num_complex::ComplexFloat;
use realfft::RealFftPlanner;

use skia_safe::gpu::direct_contexts::make_gl;
use skia_safe::gpu::gl::{FramebufferInfo, Interface};
use skia_safe::gpu::{self, backend_render_targets, SurfaceOrigin};
use skia_safe::{Color, ColorType, Paint, PaintStyle, Path, Surface};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const MARGIN_VW: f32 = 0.01;
const FG: Color = Color::BLACK;
const BG: Color = Color::YELLOW;

const SAMPLERATE: i32 = 48000;
const FFT_SIZE: usize = 2048;
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

fn hann_window(n: usize) -> impl Iterator<Item = f32> {
    (0..n).map(move |i| {
        let y: f32 = (PI * i as f32 / n as f32).sin();

        y*y
    })
}
struct AudioRecorder<const N: usize>(Box::<CircularBuffer::<N, f32>>);

impl<const N: usize> AudioRecorder<N> { 
    fn default() -> Self {
        Self (CircularBuffer::<N, f32>::boxed())
    }
}

impl<const N: usize> AudioCallback for AudioRecorder<N> {
    type Channel = f32;

    fn callback(&mut self, samples: &mut [Self::Channel]) {
        self.0.extend_from_slice(&samples);
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
    let hann_window: Vec<f32> = hann_window(FFT_SIZE as usize).collect();
    let mut windowed_signal = fft.make_input_vec();
    let mut frequency_bins = fft.make_output_vec();
    let mut gain = 18.0;
    let mut gain_multiplier = 10.0.powf(gain / 20.0);

    const BIN_WIDTH: f32 = FFT_SIZE as f32 / SAMPLERATE as f32;
    const LOW_BIN: usize = (BIN_WIDTH * MIN_FREQ as f32) as usize;
    const HIGH_BIN: usize = (BIN_WIDTH * MAX_FREQ as f32) as usize;

    let mut smoothed_fft = vec![0.0; HIGH_BIN - LOW_BIN];
    
    let mut plot = Path::new();
    let mut plot_paint = Paint::default();
    
    plot_paint
        .set_color(FG)
        .set_stroke_width(1.25)
        .set_style(PaintStyle::Stroke)
        .set_anti_alias(true);


    let mut device = sdl_audio
        .open_capture(
            None,
            &AudioSpecDesired {
                channels: Some(1),
                freq: Some(SAMPLERATE),
                samples: Some(FFT_SIZE as u16 / 2),
            },
            |_| AudioRecorder::<FFT_SIZE>::default(),
        )
        .unwrap();

    let gl_attr = sdl_video.gl_attr();
    gl_attr.set_context_profile(GLProfile::Core);
    gl_attr.set_context_version(4, 6);

    let mut window = sdl_video
        .window(&get_window_title(gain), WIDTH, HEIGHT)
        .position_centered()
        .resizable()
        .opengl()
        .build()
        .unwrap();

    // Subsequent calls to OpenGL would be performed under this context.
    // Although, this variable itself isn't of any use, its existence is.
    let _opengl_context = window.gl_create_context().unwrap();

    let load_gl_proc = |name: &str| {
        sdl_video.gl_get_proc_address(name.as_ref()) as *const _
    };

    gl::load_with(load_gl_proc);

    let interface = Interface::new_load_with(load_gl_proc).unwrap();

    let mut gr_context = make_gl(interface, None).unwrap();
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

    let mut is_hidden = false;

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
                    },
                    WindowEvent::Hidden => {
                        is_hidden = true;

                        device.pause();
                    },
                    WindowEvent::Exposed => {
                        is_hidden = false;

                        device.resume();
                    }
                    _ => {}
                },
                Event::KeyDown {
                    keycode: Some(Keycode::J),
                    ..
                } => {
                    gain += 0.1;

                    gain_multiplier = 10.0.powf(gain / 20.0);

                    window.set_title(&get_window_title(gain)).unwrap();
                }
                Event::KeyDown {
                    keycode: Some(Keycode::K),
                    ..
                } => {
                    gain -= 0.1;

                    if gain < 0.0 {
                        gain = 0.0;
                    }

                    gain_multiplier = 10.0.powf(gain / 20.0);

                    window.set_title(&get_window_title(gain)).unwrap();
                },
                _ => {}
            }
        }

        // If the window is hidden, the application should not do anything during that period.
        if is_hidden {
            break;
        }

        /* Event Loop End */

        let callback_context = device.lock();

        for (i, s) in callback_context.0.iter().enumerate() {
            windowed_signal[i] = hann_window[i] * (*s);
        }

        // An explicit drop is required because if the lock is held for too long, callback will be inhibited to recieve
        // data on time, ultimately causing horrendous lags.
        drop(callback_context);

        fft.process_with_scratch(&mut windowed_signal, &mut frequency_bins, &mut fft_scratch)
           .unwrap();

        /* Drawing calls begin */

        skia_surface.canvas().draw_color(BG, None);

        let (width, height) = (skia_surface.width() as f32, skia_surface.height() as f32);
        let margin = MARGIN_VW * width;

        let x_step = (width - 2.0 * margin) / (HIGH_BIN - LOW_BIN) as f32;
        let mut x = x_step + margin;
        let mut sign = 1.0;

        plot.move_to((margin, 0.5 * height));

        for (i, bin) in frequency_bins[LOW_BIN..HIGH_BIN].iter().enumerate() {
            let y = SMOOTHING_TIME_CONST * smoothed_fft[i]
                + (1.0 - SMOOTHING_TIME_CONST) * gain_multiplier * bin.abs() * NORMALIZATION;
            
            smoothed_fft[i] = y;
            plot.line_to((x, 0.5 * height * (1.0 - sign * y)));

            x += x_step;
            sign *= -1.0;
        }

        skia_surface.canvas().draw_path(&plot, &plot_paint);
        plot.reset();

        /* Drawing calls end */

        gr_context.flush_and_submit();

        window.gl_swap_window();
    }
}
