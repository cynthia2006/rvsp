use clap::Parser;

use sdl2::audio::AudioSpecDesired;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;

use realfft::RealFftPlanner;

use const_format::formatcp;

mod cli;
mod callback;
mod render;
mod utils;

const WINDOW_TITLE: &'static str = formatcp!(
    "{} (v{})",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_VERSION")
);

fn main() {
    let args = cli::Args::parse();

    let sdl = sdl2::init().unwrap();
    let sdl_audio = sdl.audio().unwrap();
    let sdl_video = sdl.video().unwrap();
    let mut sdl_events = sdl.event_pump().unwrap();

    let mut planner = RealFftPlanner::new();
    let fft = planner.plan_fft_forward(args.fft_size);
    let mut fft_scratch = fft.make_scratch_vec();
    let blackman_window: Vec<f32> = utils::blackman_window(args.fft_size).collect();
    let mut windowed_signal = fft.make_input_vec();
    let mut frequency_bins = fft.make_output_vec();
    
    let mut device = sdl_audio
        .open_capture(
            None,
            &AudioSpecDesired {
                channels: Some(1),
                freq: Some(48000),
                samples: Some(1024),
            },
            |_| callback::Callback {
                buffer: vec![0f32; args.fft_size]
            },
        )
        .unwrap();

    let window = sdl_video
        .window(WINDOW_TITLE, args.width, args.height)
        .position_centered()
        .resizable()
        .build()
        .unwrap();

    let canvas = window
        .into_canvas()
        .present_vsync()
        .accelerated()
        .build()
        .unwrap();

    device.resume();

    let mut renderer = render::Renderer::new(
        canvas,
        args.fft_size,
        args.tau,
        args.db_min,
        args.db_max,
        args.fg,
        args.bg,
        args.width as f32,
        args.height as f32
    );

    'running: loop {
        for event in sdl_events.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::Resized(w, h) => {
                        renderer.set_width(w as f32);
                        renderer.set_height(h as f32);
                    },
                    _ => {}
                },
                _ => {},
            }
        }

        let callback_context = device.lock();

        for (i, s) in callback_context.buffer.iter().enumerate() {
            windowed_signal[i] = blackman_window[i] * (*s);
        }

        // An explicit drop is required because if the lock is held for too long, callback will be inhibited to recieve
        // data on time, ultimately causing horrendous lags.
        drop(callback_context);

        fft.process_with_scratch(&mut windowed_signal, &mut frequency_bins, &mut fft_scratch).unwrap();

        renderer.render(&frequency_bins);
    }
}
