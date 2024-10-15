use std::f32::consts::PI;
use std::num::ParseIntError;

use skia_safe::Color;

pub(crate) fn str_to_color(s: &str) -> Result<Color, ParseIntError> {
    let c = u32::from_str_radix(s, 16)?;

    let a = c & 0xFF;
    let rgb = c >> 8;
    let argb = a << 24 | rgb;

    Ok(argb.into())
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
