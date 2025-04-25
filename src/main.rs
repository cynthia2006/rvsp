use std::cell::RefCell;
use std::f32::consts::{PI, SQRT_2};
use std::io::Cursor;
use std::mem;
use std::rc::Rc;
use std::time::Duration;

use circular_buffer::CircularBuffer;

use glium::implement_vertex;
use glium::index::{NoIndices, PrimitiveType};
use glium::uniforms::EmptyUniforms;
use glium::{DrawParameters, Program, Surface, VertexBuffer};

use pipewire as pw;
use pipewire::context::Context;
use pipewire::main_loop::MainLoop;
use pipewire::properties::properties;
use pipewire::spa::param::audio::{AudioFormat, AudioInfoRaw};
use pipewire::spa::param::ParamType;
use pipewire::spa::pod::serialize::PodSerializer;
use pipewire::spa::pod;
use pipewire::spa::pod::Pod;
use pipewire::spa::utils::{Direction, SpaTypes};
use pipewire::stream::{Stream, StreamFlags, StreamRef};

use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::video::GLProfile;

use realfft::num_complex::{Complex32, ComplexFloat};
use realfft::RealFftPlanner;

mod sdl_backend;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const MARGIN_VW: f32 = 0.01;
const LINE_WIDTH: f32 = 1.5;
const MSAA: u8 = 8; // 8x MSAA.

const SAMPLERATE: u32 = 48000;
const FFT_SIZE: usize = 2048;
const MIN_FREQ: i32 = 50;
const MAX_FREQ: i32 = 10000;
const GAIN: f32 = 23.0;
const NORMALIZATION: f32 = 1.0 / FFT_SIZE as f32;
const SMOOTHING_TIME_CONST: f32 = 0.6;

fn make_window_title(gain: f32) -> String {
    format!(
        "{} v{} ({:.2}dB)",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        gain
    )
}

fn hann_window(win: &mut [f32]) {
    const A0: f32 = 0.5;

    let n = win.len();

    for (i, w) in win.iter_mut().enumerate() {
        let x: f32 = (2.0 * PI * i as f32 / n as f32).cos();

        *w = A0 - (1.0 - A0) * x;
    }
}

#[inline(always)]
fn db_rms_to_factor(rms: f32) -> f32 {
    SQRT_2 * 10.0.powf(rms / 20.0)
}

struct AudioRecorder<const N1: usize, const N2: usize> {
    window: CircularBuffer<N1, f32>,
    channels: usize,
    buffer: [f32; N2],
}

impl<const N1: usize, const N2: usize> AudioRecorder<N1, N2> {
    fn default() -> Self {
        Self {
            window: CircularBuffer::new(),
            channels: 0,
            buffer: [0.0; N2],
        }
    }

    fn callback(&mut self, stream: &StreamRef) {
        let mut buffer = stream.dequeue_buffer().unwrap();
        let datas = buffer.datas_mut();
        if datas.is_empty() {
            return;
        }

        let data = &mut datas[0];
        let n_chans = self.channels;

        if let Some(samples) = data.data() {
            for (i, s) in self.buffer.iter_mut().enumerate() {
                *s = 0.0;

                // Downmixing multi-channel audio; PipeWire would not do it for us.
                for c in 0..n_chans {
                    let start = (n_chans * i + c) * mem::size_of::<f32>();
                    let end = start + mem::size_of::<f32>();

                    *s += f32::from_le_bytes(samples[start..end].try_into().unwrap());
                }

                *s /= n_chans as f32;
            }

            self.window.extend_from_slice(&self.buffer);
        }
    }
}

fn main() {
    pw::init();

    let sdl = sdl3::init().unwrap();
    let sdl_video = sdl.video().unwrap();
    let mut sdl_events = sdl.event_pump().unwrap();

    let mut planner = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE as usize);
    let mut fft_scratch = fft.make_scratch_vec();
    let mut window_fn = [0.0; FFT_SIZE];
    let mut windowed_signal = [0.0; FFT_SIZE];
    let mut frequency_bins = [Complex32::default(); FFT_SIZE / 2 + 1];
    let mut gain = GAIN;
    let mut gain_mult = db_rms_to_factor(gain);

    hann_window(&mut window_fn);

    const BIN_WIDTH: f32 = FFT_SIZE as f32 / SAMPLERATE as f32;
    const LOW_BIN: usize = (BIN_WIDTH * MIN_FREQ as f32) as usize;
    const HIGH_BIN: usize = (BIN_WIDTH * MAX_FREQ as f32) as usize;
    const BANDWIDTH: usize = HIGH_BIN - LOW_BIN;

    let mut smoothed_fft = vec![0.0; BANDWIDTH];

    #[derive(Copy, Clone)]
    struct Coords {
        coords: [f32; 2],
    }
    implement_vertex!(Coords, coords);

    let mut plot = [Coords { coords: [0.0, 0.0] }; BANDWIDTH + 1];

    const BUF_SIZE: usize = FFT_SIZE / 2;

    let audio_recorder = Rc::new(RefCell::new(AudioRecorder::<FFT_SIZE, BUF_SIZE>::default()));
    let audio_recorder_clone = audio_recorder.clone();

    let pw_mainloop = MainLoop::new(None).unwrap();
    let pw_context = Context::new(&pw_mainloop).unwrap();
    let pw_core = pw_context
        .connect(None)
        .expect("Could not connect to PipeWire context");

    let stream = Stream::new(
        &pw_core,
        &format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
        properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Monitor",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::NODE_LATENCY => "1024/48000",
            // Monitor system audio; the default sink.
            *pw::keys::STREAM_CAPTURE_SINK => "true",
        },
    )
    .unwrap();

    // TODO Handle other events on the stream as well, if required.
    let _listener = stream
        .add_local_listener_with_user_data(audio_recorder_clone)
        .param_changed(|_, user_data, id, param| {
            if id != ParamType::Format.as_raw() {
                return;
            }

            let mut format = AudioInfoRaw::new();
            format.parse(&param.unwrap()).unwrap();

            user_data.borrow_mut().channels = format.channels() as usize;
        })
        .process(|stream, user_data| user_data.borrow_mut().callback(&stream))
        .register()
        .unwrap();

    let raw_stream_params: Vec<u8> = PodSerializer::serialize(
        Cursor::new(Vec::new()),
        &pod::Value::Object(pod::Object {
            type_: SpaTypes::ObjectParamFormat.as_raw(),
            id: ParamType::EnumFormat.as_raw(),
            properties: {
                let mut info = AudioInfoRaw::new();

                info.set_format(AudioFormat::F32LE);
                info.set_rate(48000);

                info.into()
            },
        }),
    )
    .unwrap()
    .0
    .into_inner();

    let mut stream_params = [Pod::from_bytes(&raw_stream_params).unwrap()];

    stream
        .connect(
            Direction::Input,
            None,
            StreamFlags::AUTOCONNECT 
            | StreamFlags::MAP_BUFFERS,
            &mut stream_params,
        )
        .unwrap();

    let gl_attr = sdl_video.gl_attr();
    gl_attr.set_context_profile(GLProfile::Core);
    gl_attr.set_context_version(3, 3);
    gl_attr.set_multisample_buffers(1);
    gl_attr.set_multisample_samples(MSAA);

    let window = Rc::new(RefCell::new(
        sdl_video
            .window(&make_window_title(gain), WIDTH, HEIGHT)
            .position_centered()
            .resizable()
            .opengl()
            .build()
            .unwrap(),
    ));

    let backend = sdl_backend::SDLBackend::new(window.clone()).unwrap();
    let display = sdl_backend::Display::new(backend).unwrap();

    const VERTEX_SHADER_SRC: &str = r"#version 330 core
    in vec2 coords;

    void main() {
        gl_Position = vec4(coords.x, coords.y, 0.0f, 1.0f);
    }";

    const FRAGMENT_SHADER_SRC: &str = r"#version 330 core
    out vec4 FragColor;

    void main() {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }";

    let plot_buffer = VertexBuffer::dynamic(&display, &plot).unwrap();
    let indices = NoIndices(PrimitiveType::LineStrip);
    let program =
        Program::from_source(&display, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, None).unwrap();
    let draw_params = DrawParameters {
        line_width: Some(LINE_WIDTH),
        ..Default::default()
    };

    plot[0].coords = [MARGIN_VW - 1.0, 0.0];

    /* Preemptively generate X coordinates once, then reused by GPU infinite times. */
    const X_STEP: f32 = 2.0 * (1.0 - MARGIN_VW) / BANDWIDTH as f32;
    let mut x = X_STEP + MARGIN_VW - 1.0;

    for vertex in plot[1..].iter_mut() {
        vertex.coords[0] = x;
        x += X_STEP;
    }

    'running: loop {
        /* Event Loop Begin */
        for event in sdl_events.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::J),
                    ..
                } => {
                    gain += 0.1;

                    gain_mult = db_rms_to_factor(gain);

                    window
                        .borrow_mut()
                        .set_title(&make_window_title(gain))
                        .unwrap();
                }
                Event::KeyDown {
                    keycode: Some(Keycode::K),
                    ..
                } => {
                    gain -= 0.1;

                    gain_mult = db_rms_to_factor(gain);

                    window
                        .borrow_mut()
                        .set_title(&make_window_title(gain))
                        .unwrap();
                }
                _ => {}
            }
        }

        // NOTE Without a timeout, the PipeWire event-loop may continue indefinitely.
        pw_mainloop.loop_().iterate(Duration::from_secs(0));

        /* Event Loop End */
        let audio_recorder = audio_recorder.borrow();
        for (i, s) in audio_recorder.window.iter().enumerate() {
            windowed_signal[i] = window_fn[i] * (*s);
        }
        drop(audio_recorder);

        fft.process_with_scratch(&mut windowed_signal, &mut frequency_bins, &mut fft_scratch)
            .unwrap();

        let mut frame = display.draw();
        frame.clear_color(1.0, 1.0, 0.0, 1.0);

        let mut sign = 1.0;
        for (i, bin) in frequency_bins[LOW_BIN..HIGH_BIN].iter().enumerate() {
            let y = SMOOTHING_TIME_CONST * smoothed_fft[i]
                + (1.0 - SMOOTHING_TIME_CONST) * gain_mult * bin.abs() * NORMALIZATION;

            smoothed_fft[i] = y;
            plot[i + 1].coords[1] = sign * y;

            sign *= -1.0;
        }

        plot_buffer.write(&plot);
        frame
            .draw(
                &plot_buffer,
                &indices,
                &program,
                &EmptyUniforms,
                &draw_params,
            )
            .unwrap();
        frame.finish().unwrap();
    }
}
