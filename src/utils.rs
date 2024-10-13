use std::num::ParseIntError;
use std::f32::consts::PI;

use sdl2::{sys::SDL_FRect, pixels::Color};

pub(crate) fn str_to_color(s: &str) -> Result<Color, ParseIntError> {
    let c = u32::from_str_radix(s, 16)?;

    Ok(Color {
        a: (c >> 24) as u8,
        r: (c >> 16 & 0xFF) as u8,
        g: (c >> 8 & 0xFF) as u8,
        b: (c & 0xFF) as u8, 
    })
}

/// Generates a Blackman window of a certain length.
/// 
/// https://webaudio.github.io/web-audio-api/#blackman-window
pub(crate) fn blackman_window(n: usize) -> impl Iterator<Item = f32> {
    const A0: f32 = 0.42f32;
    const A1: f32 = 0.5f32;
    const A2: f32 = 0.08f32;

    (0..n).map(move |i| {
        let phi: f32 = 2.0 * PI * i as f32 / n as f32;

        A0 - A1 * phi.cos() + A2 * (2.0 * phi).cos()
    })
}

pub(crate) const fn frect_new() -> SDL_FRect {
    SDL_FRect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 }
}