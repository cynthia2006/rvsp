use std::f32::consts::{PI, SQRT_2};

use circular_buffer::CircularBuffer;

use sdl3::video::GLProfile;
use sdl3::audio::{AudioFormat, AudioRecordingCallback, AudioSpec, AudioStream};
use sdl3::event::{Event, WindowEvent};
use sdl3::keyboard::Keycode;

use realfft::num_complex::{Complex32, ComplexFloat};
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
const GAIN: f32 = 20.0;
const NORMALIZATION: f32 = 2.0 / FFT_SIZE as f32;
const SMOOTHING_TIME_CONST: f32 = 0.6;

fn make_window_title(gain: f32) -> String {
    format!(
        "{} v{} ({:.2}dB)",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        gain
    )
}

fn hann_window(n: usize) -> impl Iterator<Item = f32> {
    const A0: f32 = 0.5;

    (0..n).map(move |i| {
        let y: f32 = (2.0 * PI * i as f32 / n as f32).cos();

        A0 - (1.0 - A0) * y
    })
}

#[inline(always)]
fn db_rms_to_factor(rms: f32) -> f32 {
    SQRT_2 * 10.0.powf(rms / 20.0)
}

struct AudioRecorder<const N1: usize, const N2: usize> {
    window: Box::<CircularBuffer::<N1, f32>>,
    buffer: Box::<[f32; N2]>,
    buf_pos: usize,
}

impl<const N1: usize, const N2: usize> AudioRecorder<N1, N2> {
    fn default() -> Self {
        Self {
            window: CircularBuffer::boxed(),
            buffer: Box::new([0.0; N2]),
            buf_pos: 0,
        }
    }
}

impl<const N1: usize, const N2: usize> AudioRecordingCallback<f32> for AudioRecorder<N1, N2> {
    fn callback(&mut self, stream: &mut AudioStream, _available: i32) {
        let buf_len = self.buffer.len();
        let samples_read = stream.read_f32_samples(&mut self.buffer[self.buf_pos..]).unwrap();

        self.buf_pos = (self.buf_pos + samples_read) % buf_len;
        
        // Only push-back if complete (ensures 50% overlap).
        if self.buf_pos == 0 {
            self.window.extend_from_slice(&*self.buffer);
        }
    }
}

fn main() {
    let sdl = sdl3::init().unwrap();
    let sdl_audio = sdl.audio().unwrap();
    let sdl_video = sdl.video().unwrap();
    let mut sdl_events = sdl.event_pump().unwrap();

    let mut planner = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE as usize);
    let mut fft_scratch = fft.make_scratch_vec();
    let window_fn: Vec<f32> = hann_window(FFT_SIZE as usize).collect();
    let mut windowed_signal = Box::new([0.0; FFT_SIZE]);
    let mut frequency_bins = Box::new([Complex32::default(); FFT_SIZE / 2 + 1]);
    let mut gain = GAIN;
    let mut gain_multiplier = db_rms_to_factor(gain);

    const BIN_WIDTH: f32 = FFT_SIZE as f32 / SAMPLERATE as f32;
    const LOW_BIN: usize = (BIN_WIDTH * MIN_FREQ as f32) as usize;
    const HIGH_BIN: usize = (BIN_WIDTH * MAX_FREQ as f32) as usize;

    let mut smoothed_fft = vec![0.0; HIGH_BIN - LOW_BIN];

    let mut plot = Path::new();
    let mut plot_paint = Paint::default();

    plot_paint
        .set_color(FG)
        .set_stroke_width(0.0025)
        .set_style(PaintStyle::Stroke)
        .set_anti_alias(true);

    const BUF_SIZE: usize = FFT_SIZE / 2;        
    let mut stream = sdl_audio
        .open_recording_stream(
            &AudioSpec {
                channels: Some(1),
                freq: Some(SAMPLERATE),
                format: Some(AudioFormat::f32_sys())
            },
            AudioRecorder::<FFT_SIZE, BUF_SIZE>::default(),
        )
        .unwrap();

    let gl_attr = sdl_video.gl_attr();
    gl_attr.set_context_profile(GLProfile::Core);
    gl_attr.set_context_version(4, 6);

    let mut window = sdl_video
        .window(&make_window_title(gain), WIDTH, HEIGHT)
        .position_centered()
        .resizable()
        .opengl()
        .build()
        .unwrap();

    // Subsequent calls to OpenGL would be performed under this context.
    // Although, this variable itself isn't of any use, its existence is.
    let _opengl_context = window.gl_create_context().unwrap();

    window.gl_set_context_to_current().unwrap();

    let mut gr_context = make_gl(Interface::new_native().unwrap(), None).unwrap();

    fn create_surface(
        width: i32,
        height: i32,
        gr_context: &mut skia_safe::gpu::DirectContext,
    ) -> Option<Surface> {
        let fb_info = FramebufferInfo {
            format: skia_safe::gpu::gl::Format::RGBA8.into(),
            ..Default::default()
        };

        let backend_render_target =
            backend_render_targets::make_gl((width, height), None, 0, fb_info);

        gpu::surfaces::wrap_backend_render_target(
            gr_context,
            &backend_render_target,
            SurfaceOrigin::BottomLeft,
            ColorType::N32,
            None,
            None,
        )
        .map(|mut s| {
            /* This is so that, OpenGL NDC coordinates are emulated. */
            s.canvas().scale((width as f32 / 2.0, -height as f32 / 2.0));
            s.canvas().translate((1.0, -1.0));

            s
        })
    }

    let mut skia_surface = create_surface(WIDTH as i32, HEIGHT as i32, &mut gr_context).unwrap();

    stream.resume().unwrap();

    'running: loop {
        /* Event Loop Begin */
        for event in sdl_events.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::Window { win_event: WindowEvent::Resized(w, h), .. } => {
                    skia_surface = create_surface(w as i32, h as i32, &mut gr_context).unwrap()
                },
                Event::KeyDown {
                    keycode: Some(Keycode::J),
                    ..
                } => {
                    gain += 0.1;

                    gain_multiplier = db_rms_to_factor(gain);

                    window.set_title(&make_window_title(gain)).unwrap();
                }
                Event::KeyDown {
                    keycode: Some(Keycode::K),
                    ..
                } => {
                    gain -= 0.1;

                    gain_multiplier = db_rms_to_factor(gain);

                    window.set_title(&make_window_title(gain)).unwrap();
                }
                _ => {}
            }
        }

        /* Event Loop End */
        let callback_context = stream.lock().unwrap();

        for (i, s) in callback_context.window.iter().enumerate() {
            windowed_signal[i] = window_fn[i] * (*s);
        }
        
        // An explicit drop is required because if the lock is held for too long, callback will be inhibited to recieve
        // data on time, ultimately causing horrendous lags.
        drop(callback_context);

        fft.process_with_scratch(&mut *windowed_signal, &mut *frequency_bins, &mut fft_scratch)
            .unwrap();

        /* Drawing calls begin */
        skia_surface.canvas().draw_color(BG, None);

        const X_STEP: f32 = 2.0 * (1.0 - MARGIN_VW) / (HIGH_BIN - LOW_BIN) as f32;
        let mut x = X_STEP + MARGIN_VW - 1.0;
        let mut sign = 1.0;

        plot.move_to((MARGIN_VW - 1.0, 0.0));

        for (i, bin) in frequency_bins[LOW_BIN..HIGH_BIN].iter().enumerate() {
            let y = SMOOTHING_TIME_CONST * smoothed_fft[i]
                + (1.0 - SMOOTHING_TIME_CONST) * gain_multiplier * bin.abs() * NORMALIZATION;

            smoothed_fft[i] = y;
            plot.line_to((x, sign * y));

            x += X_STEP;
            sign *= -1.0;
        }

        skia_surface.canvas().draw_path(&plot, &plot_paint);
        plot.rewind();
        /* Drawing calls end */

        gr_context.flush_and_submit();
        window.gl_swap_window();
    }
}
