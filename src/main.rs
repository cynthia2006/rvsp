use std::cell::RefCell;
use std::f32::consts::{PI, SQRT_2};
use std::io::Cursor;
use std::mem;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use glium::backend::glutin;
use glium::backend::glutin::Display;
use glium::glutin::config::ConfigTemplateBuilder;
use glium::glutin::surface::WindowSurface;
use glium::implement_vertex;
use glium::index::{NoIndices, PrimitiveType};
use glium::uniforms::EmptyUniforms;
use glium::winit::application::ApplicationHandler;
use glium::winit::dpi::PhysicalSize;
use glium::winit::event::{KeyEvent, WindowEvent};
use glium::winit::event_loop::{ActiveEventLoop, EventLoop};
use glium::winit::keyboard::{KeyCode, PhysicalKey};
use glium::winit::window::{Window, WindowId};
use glium::{DrawParameters, Program, Surface, VertexBuffer};

use lazy_static::lazy_static;

use pipewire as pw;
use pipewire::context::Context;
use pipewire::main_loop::MainLoop;
use pipewire::properties::properties;
use pipewire::spa::param::audio::{AudioFormat, AudioInfoRaw};
use pipewire::spa::param::ParamType;
use pipewire::spa::pod;
use pipewire::spa::pod::serialize::PodSerializer;
use pipewire::spa::pod::Pod;
use pipewire::spa::utils::{Direction, SpaTypes};
use pipewire::stream::{Stream, StreamFlags, StreamRef};

use realfft::num_complex::{Complex32, ComplexFloat};
use realfft::{RealFftPlanner, RealToComplex};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const MARGIN_VW: f32 = 0.01;
const LINE_WIDTH: f32 = 1.5;
const MSAA: u8 = 8; // 8x MSAA.

const SAMPLERATE: u32 = 48000;
const WINDOW_SIZE: usize = 2048;
const FFT_SIZE: usize = WINDOW_SIZE / 2 + 1;
const MIN_FREQ: i32 = 50;
const MAX_FREQ: i32 = 10000;
const GAIN: f32 = 26.0;
const NORMALIZATION: f32 = 1.0 / WINDOW_SIZE as f32;
const SMOOTHING_TIME_CONST: f32 = 0.6;

const BIN_WIDTH: f32 = WINDOW_SIZE as f32 / SAMPLERATE as f32;
const LOW_BIN: usize = (BIN_WIDTH * MIN_FREQ as f32) as usize;
const HIGH_BIN: usize = (BIN_WIDTH * MAX_FREQ as f32) as usize;
const BANDWIDTH: usize = HIGH_BIN - LOW_BIN;
const BANDWIDTH_PLUS_ONE: usize = BANDWIDTH + 1;

lazy_static! {
    static ref WINDOW_FN: [f32; WINDOW_SIZE] = (|| {
        let mut win = [0.0; WINDOW_SIZE];

        const A0: f32 = 0.5;
        for (i, w) in win.iter_mut().enumerate() {
            let x: f32 = (2.0 * PI * i as f32 / WINDOW_SIZE as f32).cos();

            *w = A0 - (1.0 - A0) * x;
        }

        win
    })();
}

/// A circular buffer, but for a specific purpose.
struct SlidingWindow<T, const N: usize> {
    buffer: [T; N],
    write_pos: usize,
}

impl<T, const N: usize> SlidingWindow<T, N> {
    fn put(&mut self, elem: T) {
        self.buffer[self.write_pos] = elem;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
    }
}

struct AudioRecorder {
    window: SlidingWindow<f32, WINDOW_SIZE>,
    channels: usize,
}

impl AudioRecorder {
    fn new() -> Self {
        Self {
            window: SlidingWindow {
                buffer: [0.0; WINDOW_SIZE],
                write_pos: 0,
            },
            channels: 0,
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
        let n_samples = data.chunk().size() as usize / mem::size_of::<f32>() / n_chans;

        if let Some(samples) = data.data() {
            for i in 0..n_samples {
                let mut s = 0.0;

                // Downmixing multi-channel audio; PipeWire would not do it for us.
                for c in 0..n_chans {
                    let start = (n_chans * i + c) * mem::size_of::<f32>();
                    let end = start + mem::size_of::<f32>();

                    s += f32::from_le_bytes(samples[start..end].try_into().unwrap());
                }

                s /= n_chans as f32;

                self.window.put(s);
            }
        }
    }
}

#[inline(always)]
fn db_rms_to_factor(rms: f32) -> f32 {
    SQRT_2 * 10.0.powf(rms / 20.0)
}

#[derive(Copy, Clone)]
struct Coords {
    coords: [f32; 2],
}
implement_vertex!(Coords, coords);

const VERTEX_SHADER_SRC: &str = r"#version 330 core
in vec2 coords;

void main() {
    gl_Position = vec4(coords.xy, 0.0f, 1.0f);
}";

const FRAGMENT_SHADER_SRC: &str = r"#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}";

struct App<'a> {
    window: Window,
    display: Display<WindowSurface>,
    pw_mainloop: MainLoop,
    audio_rec: Rc<RefCell<AudioRecorder>>,
    fft: Arc<dyn RealToComplex<f32>>,

    plot: [Coords; BANDWIDTH_PLUS_ONE],
    plot_buffer: VertexBuffer<Coords>,
    indices: NoIndices,
    program: Program,
    draw_params: DrawParameters<'a>,

    gain: f32,
    gain_multiplier: f32,
    frequency_bins: [Complex32; FFT_SIZE],
    fft_scratch: Vec<Complex32>,
    smoothed_fft: [f32; BANDWIDTH],
    windowed_signal: [f32; WINDOW_SIZE],
}

impl<'a> App<'a> {
    fn increase_gain(&mut self, increment: f32) {
        self.gain += increment;
        self.gain_multiplier = db_rms_to_factor(self.gain);
    }

    fn decrease_gain(&mut self, decrement: f32) {
        self.gain -= decrement;
        self.gain_multiplier = db_rms_to_factor(self.gain);
    }

    fn update_window_title(&self) {
        let title = format!(
            "{} v{} ({:.2}dB)",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            self.gain
        );

        self.window.set_title(&title);
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        // TODO
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyJ),
                        ..
                    },
                ..
            } => {
                self.increase_gain(0.1);
                self.update_window_title();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyK),
                        ..
                    },
                ..
            } => {
                self.decrease_gain(0.1);
                self.update_window_title();
            }
            WindowEvent::Resized(PhysicalSize { width, height }) => {
                self.display.resize((width, height));
            }
            WindowEvent::RedrawRequested => {
                let audio_recorder = self.audio_rec.borrow();
                for (i, s) in audio_recorder.window.buffer.iter().enumerate() {
                    self.windowed_signal[i] = WINDOW_FN[i] * (*s);
                }
                drop(audio_recorder);

                self.fft
                    .process_with_scratch(
                        &mut self.windowed_signal,
                        &mut self.frequency_bins,
                        &mut self.fft_scratch,
                    )
                    .unwrap();

                let mut frame = self.display.draw();
                frame.clear_color(1.0, 1.0, 0.0, 1.0);

                let mut sign = 1.0;
                for (i, bin) in self.frequency_bins[LOW_BIN..HIGH_BIN].iter().enumerate() {
                    let y = SMOOTHING_TIME_CONST * self.smoothed_fft[i]
                        + (1.0 - SMOOTHING_TIME_CONST)
                            * self.gain_multiplier
                            * bin.abs()
                            * NORMALIZATION;

                    self.smoothed_fft[i] = y;
                    self.plot[i + 1].coords[1] = sign * y;

                    sign *= -1.0;
                }

                self.plot_buffer.write(&self.plot);
                frame
                    .draw(
                        &self.plot_buffer,
                        &self.indices,
                        &self.program,
                        &EmptyUniforms,
                        &self.draw_params,
                    )
                    .unwrap();

                self.window.pre_present_notify();
                frame.finish().unwrap();
                self.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.pw_mainloop.loop_().iterate(Duration::from_millis(0));
    }
}

fn main() {
    pw::init();

    let audio_rec = Rc::new(RefCell::new(AudioRecorder::new()));
    let audio_rec_clone = audio_rec.clone();

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
        .add_local_listener_with_user_data(audio_rec_clone)
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
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
            &mut stream_params,
        )
        .unwrap();

    let winit_mainloop = EventLoop::new().unwrap();
    let (window, display) = glutin::SimpleWindowBuilder::new()
        .with_config_template_builder(ConfigTemplateBuilder::new().with_multisampling(MSAA))
        .with_inner_size(WIDTH, HEIGHT)
        .build(&winit_mainloop);

    let mut plot = [Coords { coords: [0.0, 0.0] }; BANDWIDTH_PLUS_ONE];
    plot[0].coords = [MARGIN_VW - 1.0, 0.0];

    // Generate X-coordinates beforehand.
    const X_STEP: f32 = 2.0 * (1.0 - MARGIN_VW) / BANDWIDTH as f32;
    let mut x = X_STEP + MARGIN_VW - 1.0;

    for vertex in plot[1..].iter_mut() {
        vertex.coords[0] = x;
        x += X_STEP;
    }

    let plot_buffer = VertexBuffer::dynamic(&display, &plot).unwrap();
    let indices = NoIndices(PrimitiveType::LineStrip);
    let program =
        Program::from_source(&display, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, None).unwrap();
    let draw_params = DrawParameters {
        line_width: Some(LINE_WIDTH),
        ..Default::default()
    };

    let mut planner = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE as usize);
    let frequency_bins = [Complex32::default(); FFT_SIZE];
    let fft_scratch = fft.make_scratch_vec();

    let mut app = App {
        window,
        display,
        pw_mainloop,
        audio_rec,
        fft,
        plot,
        plot_buffer,
        indices,
        program,
        draw_params,
        gain: GAIN,
        gain_multiplier: db_rms_to_factor(GAIN),
        frequency_bins,
        fft_scratch,
        smoothed_fft: [0.0; BANDWIDTH],
        windowed_signal: [0.0; WINDOW_SIZE],
    };

    app.update_window_title();

    winit_mainloop.run_app(&mut app).unwrap();
}
