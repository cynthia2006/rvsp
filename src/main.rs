use std::f32::consts::{PI, SQRT_2};
use std::ffi::CString;
use std::ptr;

use circular_buffer::CircularBuffer;

use sdl3::audio::{AudioFormat, AudioRecordingCallback, AudioSpec, AudioStream};
use sdl3::event::{Event, WindowEvent};
use sdl3::keyboard::Keycode;
use sdl3::video::GLProfile;

use realfft::num_complex::{Complex32, ComplexFloat};
use realfft::RealFftPlanner;

use gl;

use lazy_static::lazy_static;

mod utils;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const MARGIN_VW: f32 = 0.01;
const LINE_WIDTH: f32 = 1.5;

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
    window: Box<CircularBuffer<N1, f32>>,
    buffer: Box<[f32; N2]>,
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
        let samples_read = stream
            .read_f32_samples(&mut self.buffer[self.buf_pos..])
            .unwrap();

        self.buf_pos = (self.buf_pos + samples_read) % buf_len;

        // Only push-back if complete (ensures 50% overlap).
        if self.buf_pos == 0 {
            self.window.extend_from_slice(&*self.buffer);
        }
    }
}

lazy_static! {
    static ref VERTEX_SHADER_SRC: CString = CString::new(
        r"#version 330 core
    layout (location = 0) in float x;
    layout (location = 1) in float y;

    void main() {
        gl_Position = vec4(x, y, 0.0f, 1.0f);
    }
    "
    )
    .unwrap();
    static ref FRAGMENT_SHADER_SRC: CString = CString::new(
        r"#version 330 core
    out vec4 FragColor;

    void main() {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }"
    )
    .unwrap();
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
    let mut gain_mult = db_rms_to_factor(gain);

    const BIN_WIDTH: f32 = FFT_SIZE as f32 / SAMPLERATE as f32;
    const LOW_BIN: usize = (BIN_WIDTH * MIN_FREQ as f32) as usize;
    const HIGH_BIN: usize = (BIN_WIDTH * MAX_FREQ as f32) as usize;
    const BANDWIDTH: usize = HIGH_BIN - LOW_BIN;

    let mut smoothed_fft = vec![0.0; BANDWIDTH];
    let mut plot = [0f32; BANDWIDTH + 1];

    const BUF_SIZE: usize = FFT_SIZE / 2;
    let mut stream = sdl_audio
        .open_recording_stream(
            &AudioSpec {
                channels: Some(1),
                freq: Some(SAMPLERATE),
                format: Some(AudioFormat::f32_sys()),
            },
            AudioRecorder::<FFT_SIZE, BUF_SIZE>::default(),
        )
        .unwrap();

    let gl_attr = sdl_video.gl_attr();
    gl_attr.set_context_profile(GLProfile::Core);
    gl_attr.set_context_version(3, 3);
    gl_attr.set_multisample_buffers(1);
    gl_attr.set_multisample_samples(2);

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

    gl::load_with(|name| sdl_video.gl_get_proc_address(name).unwrap() as *const _);

    let mut vao = 0;
    let mut vbo_x = 0;
    let mut vbo_y = 0;
    let program = unsafe { gl::CreateProgram() };

    unsafe {
        let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
        let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);

        gl::ShaderSource(vertex_shader, 1, &VERTEX_SHADER_SRC.as_ptr(), ptr::null());
        gl::CompileShader(vertex_shader);
        utils::ensure_shader_compilation(vertex_shader).expect("Vertex shader compilation failed");

        gl::ShaderSource(
            fragment_shader,
            1,
            &FRAGMENT_SHADER_SRC.as_ptr(),
            ptr::null(),
        );
        gl::CompileShader(fragment_shader);
        utils::ensure_shader_compilation(fragment_shader)
            .expect("Fragment shader compilation failed");

        gl::AttachShader(program, vertex_shader);
        gl::AttachShader(program, fragment_shader);
        gl::LinkProgram(program);
        utils::ensure_shader_linking(program).expect("Shader linking failed");

        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);
    }

    /* Generate X coordinates once; reused by GPU infinite times. */
    plot[0] = MARGIN_VW - 1.0;

    const X_STEP: f32 = 2.0 * (1.0 - MARGIN_VW) / BANDWIDTH as f32;
    let mut x = X_STEP + MARGIN_VW - 1.0;
    
    for x_coord in plot[1..].iter_mut() {
        *x_coord = x;
        x += X_STEP;
    }

    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo_x);
        gl::GenBuffers(1, &mut vbo_y);

        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_x);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (plot.len() * size_of::<f32>()) as isize,
            plot.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::VertexAttribPointer(
            0,
            1,
            gl::FLOAT,
            gl::FALSE,
            size_of::<f32>() as i32,
            0 as *const _,
        );
        gl::EnableVertexAttribArray(0);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_y);
        gl::VertexAttribPointer(
            1,
            1,
            gl::FLOAT,
            gl::FALSE,
            size_of::<f32>() as i32,
            0 as *const _,
        );
        gl::EnableVertexAttribArray(1);

        gl::LineWidth(LINE_WIDTH);
        gl::UseProgram(program);
    }

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
                    unsafe {
                        gl::Viewport(0, 0, w, h);
                    }
                },
                Event::KeyDown {
                    keycode: Some(Keycode::J),
                    ..
                } => {
                    gain += 0.1;

                    gain_mult = db_rms_to_factor(gain);

                    window.set_title(&make_window_title(gain)).unwrap();
                }
                Event::KeyDown {
                    keycode: Some(Keycode::K),
                    ..
                } => {
                    gain -= 0.1;

                    gain_mult = db_rms_to_factor(gain);

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
        drop(callback_context);

        fft.process_with_scratch(
            &mut *windowed_signal,
            &mut *frequency_bins,
            &mut fft_scratch,
        )
        .unwrap();

        /* Drawing calls begin */
        unsafe {
            gl::ClearColor(1.0, 1.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        let mut sign = 1.0;

        plot[0] = 0.0;
        for (i, bin) in frequency_bins[LOW_BIN..HIGH_BIN].iter().enumerate() {
            let y = SMOOTHING_TIME_CONST * smoothed_fft[i]
                + (1.0 - SMOOTHING_TIME_CONST) * gain_mult * bin.abs() * NORMALIZATION;

            smoothed_fft[i] = y;
            plot[i + 1] = sign * y;

            sign *= -1.0;
        }

        unsafe {
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (plot.len() * size_of::<f32>()) as isize,
                plot.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );
            gl::DrawArrays(gl::LINE_STRIP, 0, BANDWIDTH as i32);
        }

        window.gl_swap_window();
    }
}
