use clap::Parser;

use sdl2::pixels::Color;

use crate::utils;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    /// FFT size (in samples)
    #[arg(short = 's', long, default_value = "1024")]
    pub fft_size: usize,

    /// Initial width of window
    #[arg(long, default_value = "800")]
    pub width: u32,

    /// Initial height of window
    #[arg(long, default_value = "600")]
    pub height: u32,

    /// Smoothing time constant (0..1)
    #[arg(long, default_value = "0.3")]
    pub tau: f32,

    /// Clamping minimum (in decibels, -∞..0)
    #[arg(long, default_value = "-80")]
    pub db_min: f32,

    /// Clamping maxmimum (in decibels, -∞..0)
    #[arg(long, default_value = "-20")]
    pub db_max: f32,

    /// Foreground color (in #AARRGGBB format)
    #[arg(long, default_value = "ffffffff", value_parser = utils::str_to_color)]
    pub fg: Color,

    /// Background color (in #AARRGGBB format)
    #[arg(long, default_value = "ff000000", value_parser = utils::str_to_color)]
    pub bg: Color
}
