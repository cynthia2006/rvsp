use std::cell::RefCell;
use std::f32::consts::{PI, SQRT_2};
use std::ffi::CString;
use std::io::Cursor;
use std::marker::PhantomPinned;
use std::mem::{size_of, MaybeUninit};
use std::num::NonZeroU32;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextAttributesBuilder, PossiblyCurrentContext};
use glutin::display::GetGlDisplay;
use glutin::prelude::{GlDisplay, NotCurrentGlContext, PossiblyCurrentGlContext};
use glutin::surface::{GlSurface, Surface, SwapInterval, WindowSurface};

use glutin_winit::GlWindow;

use winit::application::ApplicationHandler;
use winit::dpi::{self, PhysicalSize};
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::raw_window_handle::HasWindowHandle;
use winit::window::{Window, WindowId};

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
use pipewire::spa::support::system::IoFlags;
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
const WINDOW_SIZE_HALF: usize = WINDOW_SIZE / 2;
const FFT_SIZE: usize = WINDOW_SIZE / 2 + 1;
const MIN_FREQ: i32 = 50;
const MAX_FREQ: i32 = 10000;
const GAIN: f32 = 26.0;
const NORMALIZATION: f32 = 1.0 / WINDOW_SIZE as f32;
const SMOOTHING_FACTOR: f32 = 0.6;

const BIN_WIDTH: f32 = WINDOW_SIZE as f32 / SAMPLERATE as f32;
const LOW_BIN: usize = (BIN_WIDTH * MIN_FREQ as f32) as usize;
const HIGH_BIN: usize = (BIN_WIDTH * MAX_FREQ as f32) as usize;
const BANDWIDTH: usize = HIGH_BIN - LOW_BIN;
const BANDWIDTH_PLUS_ONE: usize = BANDWIDTH + 1;

mod sliding_window;
mod polygon_renderer;

use polygon_renderer::PolygonRenderer;
use sliding_window::SlidingWindowAdapter;

lazy_static! {
    // Hann window function.
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

struct AudioRecorder<'a> {
    window_buffer: [f32; WINDOW_SIZE],
    window: MaybeUninit<SlidingWindowAdapter<'a, f32>>,
    channels: usize,
    _pinned: PhantomPinned,
}

impl<'a> AudioRecorder<'a> {
    // NOTE A Pin<Box<T>> inside a Rc<T> is redundant because Rc<T> allocates its data on heap. 
    // Unlike Pin<Box<T>>, Pin<Rc<T>> doesn't have a Pin<T>::get_mut() method because a Rc<T>
    // hands out immutable references; RefCell<T> enables interior mutability in this case.
    fn new() -> Pin<Rc<RefCell<Self>>> {
        let buffer = [0.0; WINDOW_SIZE];
        let recorder = Rc::pin(RefCell::new(Self {
            window_buffer: buffer,
            window: MaybeUninit::uninit(),
            channels: 0,
            // Mark it as !Unpin.
            _pinned: PhantomPinned
        }));

        unsafe {
            let mut recorder_ref = recorder.borrow_mut();
            // Borrow rules don't cover self-referential structs, so we create a raw pointer to evade it.
            let buffer_ptr: *mut [f32] = &mut recorder_ref.window_buffer;

            recorder_ref.window.write(SlidingWindowAdapter::new(&mut *buffer_ptr, 0));
        };

        recorder
    }

    fn callback(&mut self, stream: &StreamRef) {
        let mut buffer = stream.dequeue_buffer().unwrap();
        let datas = buffer.datas_mut();
        if datas.is_empty() {
            return;
        }

        let data = &mut datas[0];
        let n_chans = self.channels;
        let n_samples = data.chunk().size() as usize / size_of::<f32>() / n_chans;

        if let Some(samples) = data.data() {
            for i in 0..n_samples {
                let mut s = 0.0;

                // Downmixing multi-channel audio; PipeWire would not do it for us.
                for c in 0..n_chans {
                    let start = (n_chans * i + c) * size_of::<f32>();
                    let end = start + size_of::<f32>();

                    s += f32::from_le_bytes(samples[start..end].try_into().unwrap());
                }

                s /= n_chans as f32;

                unsafe {
                    self.window.assume_init_mut().put(s);
                }
            }
        }
    }
}

#[inline(always)]
fn db_rms_to_factor(rms: f32) -> f32 {
    SQRT_2 * 10.0.powf(rms / 20.0)
}

struct App<'a> {
    stream: Box<Stream>,

    window: Window,
    gl_context: PossiblyCurrentContext,
    gl_surface: Surface<WindowSurface>,
    audio_rec: Pin<Rc<RefCell<AudioRecorder<'a>>>>,
    fft: Arc<dyn RealToComplex<f32>>,

    plot: [f32; 2*BANDWIDTH_PLUS_ONE],
    renderer: PolygonRenderer,

    gain: f32,
    gain_multiplier: f32,
    smoothing_factor: f32,
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

    fn increase_smoothing(&mut self, increment: f32) {
        self.smoothing_factor += increment;
        self.smoothing_factor = self.smoothing_factor.clamp(0.0, 1.0);
    }

    fn decrease_smoothing(&mut self, decrement: f32) {
        self.smoothing_factor -= decrement;
        self.smoothing_factor = self.smoothing_factor.clamp(0.0, 1.0);
    }

    fn update_window_title(&self) {
        let title = format!(
            "{} v{} (🔊 {:.2}dB , τ = {:.2})",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            self.gain,
            self.smoothing_factor
        );

        self.window.set_title(&title);
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        self.stream.set_active(true).unwrap();
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
                        physical_key: PhysicalKey::Code(KeyCode::ArrowUp),
                        state: ElementState::Pressed,
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
                        physical_key: PhysicalKey::Code(KeyCode::ArrowDown),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.decrease_gain(0.1);
                self.update_window_title();
            },
            WindowEvent::KeyboardInput { 
                event: KeyEvent { 
                    physical_key: PhysicalKey::Code(KeyCode::ArrowLeft),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                self.decrease_smoothing(0.01);
                self.update_window_title();
            },
            WindowEvent::KeyboardInput { 
                event: KeyEvent { 
                    physical_key: PhysicalKey::Code(KeyCode::ArrowRight),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                self.increase_smoothing(0.01);
                self.update_window_title();
            },

            WindowEvent::Resized(PhysicalSize { width, height }) => {
                self.gl_surface.resize(
                    &self.gl_context, 
                    NonZeroU32::new(width).unwrap(),
                    NonZeroU32::new(height).unwrap());
            },
            WindowEvent::RedrawRequested => {
                let audio_recorder = self.audio_rec.borrow();
                let window = unsafe { audio_recorder.window.assume_init_ref() };
                for (i, s) in window.iter().enumerate() {
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

                unsafe { self.renderer.clear((1.0, 1.0, 0.0)); }

                let mut sign = 1.0;
                for (i, bin) in self.frequency_bins[LOW_BIN..HIGH_BIN].iter().enumerate() {
                    let y = self.smoothing_factor * self.smoothed_fft[i]
                        + (1.0 - self.smoothing_factor)
                            * self.gain_multiplier
                            * bin.abs()
                            * NORMALIZATION;

                    self.smoothed_fft[i] = y;
                    self.plot[2*(i + 1) + 1] = sign * y;

                    sign *= -1.0;
                }

                unsafe { self.renderer.upload(&self.plot); }

                self.window.pre_present_notify();
                self.gl_surface.swap_buffers(&self.gl_context).unwrap();
                self.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    pw::init();

    let audio_rec = AudioRecorder::new();

    let pw_mainloop = MainLoop::new(None).unwrap();
    let pw_context = Context::new(&pw_mainloop).unwrap();
    let pw_core = pw_context
        .connect(None)
        .expect("Could not connect to PipeWire context");

    let stream = Box::new(
        Stream::new(
            &pw_core,
            &format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
            properties! {
                *pw::keys::MEDIA_TYPE => "Audio",
                *pw::keys::MEDIA_CATEGORY => "Monitor",
                *pw::keys::MEDIA_ROLE => "Music",
                *pw::keys::NODE_LATENCY => format!("{}/{}", WINDOW_SIZE_HALF, SAMPLERATE),
                // Monitor system audio; the default sink.
                *pw::keys::STREAM_CAPTURE_SINK => "true",
            },
        )
        .unwrap(),
    );

    // TODO Handle other events on the stream as well, if required.
    let _listener = stream
        .add_local_listener_with_user_data(audio_rec.clone())
        .param_changed(|_, user_data, id, param| {
            if id != ParamType::Format.as_raw() {
                return;
            }

            // Channels would be zero, when the format is cleared.
            let mut channels = 0;

            if let Some(param) = param {
                let mut format = AudioInfoRaw::new();
                format.parse(param).unwrap();

                channels = format.channels() as usize;
            }

            user_data.borrow_mut().channels = channels;
        })
        .process(|stream, user_data| {
            user_data.as_ref().borrow_mut().callback(stream);
        })
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
                info.set_rate(SAMPLERATE);

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
            StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::INACTIVE,
            &mut stream_params,
        )
        .unwrap();

    let winit_mainloop = EventLoop::new().unwrap();
    let window_attribs = Window::default_attributes()
        .with_inner_size(dpi::PhysicalSize { width: WIDTH, height: HEIGHT });
    let gl_config_template_builder = ConfigTemplateBuilder::new()
        .with_multisampling(MSAA);

    let (window, gl_config) = glutin_winit::DisplayBuilder::new()
        .with_window_attributes(Some(window_attribs))
        .build(&winit_mainloop, gl_config_template_builder, 
            |mut configs| configs.next().unwrap())
        .map(|res| (res.0.unwrap(), res.1))
        .unwrap();

    let context_attribs = ContextAttributesBuilder::new()
        .build(window.window_handle().ok().map(|wh| wh.as_raw()));
    let gl_display = gl_config.display();
    let gl_context = unsafe { 
        gl_display.create_context(&gl_config, &context_attribs)
            .unwrap()
            .treat_as_possibly_current() 
    };

    let suface_attribs = window
        .build_surface_attributes(Default::default())
        .unwrap();
    let gl_surface = unsafe { gl_config
        .display()
        .create_window_surface(&gl_config, &suface_attribs)
        .unwrap()
    };

    gl_context.make_current(&gl_surface).unwrap();
    gl_surface.set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap())).unwrap();

    gl::load_with(|proc| gl_display.get_proc_address(CString::new(proc).unwrap().as_c_str()));

    let mut plot = [0.0; 2*BANDWIDTH_PLUS_ONE];
    plot[0] = MARGIN_VW - 1.0;
    plot[1] = 0.0;

    // Generate X-coordinates beforehand.
    const X_STEP: f32 = 2.0 * (1.0 - MARGIN_VW) / BANDWIDTH as f32;
    let mut x = X_STEP + MARGIN_VW - 1.0;

    for i in 1..BANDWIDTH_PLUS_ONE {
        plot[2*i + 0] = x;
        x += X_STEP;
    }

    let renderer = unsafe { PolygonRenderer::new(LINE_WIDTH) };

    let mut planner = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE as usize);
    let frequency_bins = [Complex32::default(); FFT_SIZE];
    let fft_scratch = fft.make_scratch_vec();

    let app = App {
        stream,
        window,
        gl_context,
        gl_surface,
        audio_rec,
        fft,
        plot,
        renderer,
        gain: GAIN,
        gain_multiplier: db_rms_to_factor(GAIN),
        smoothing_factor: SMOOTHING_FACTOR,
        frequency_bins,
        fft_scratch,
        smoothed_fft: [0.0; BANDWIDTH],
        windowed_signal: [0.0; WINDOW_SIZE],
    };

    app.update_window_title();

    let app = RefCell::new(app);
    let pw_mainloop_rc = Rc::new(pw_mainloop);
    let pw_mainloop_clone = Rc::clone(&pw_mainloop_rc);

    let _winit_source =
        pw_mainloop_rc
            .loop_()
            .add_io(winit_mainloop, IoFlags::IN | IoFlags::ERR, move |loop_| {
                if let PumpStatus::Exit(_) = loop_.pump_app_events(Some(Duration::ZERO), &mut *app.borrow_mut()) {
                    pw_mainloop_clone.quit();
                }
            });

    pw_mainloop_rc.run();
}
