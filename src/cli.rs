use clap::Parser;

use skia_safe::Color;

use crate::utils;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    /// Initial width of window
    #[arg(long, default_value = "800")]
    pub(crate) width: u32,

    /// Initial height of window
    #[arg(long, default_value = "600")]
    pub(crate) height: u32,

    /// Smoothing time constant (0..1)
    #[arg(long, default_value = "0.3")]
    pub(crate) tau: f32,

    /// Minimum frequency to consider
    #[arg(long, default_value = "50")]
    pub(crate) min_frequency: f32,

    /// Maximum frequency to consider
    #[arg(long, default_value = "10000")]
    pub(crate) max_frequency: f32,

    /// Foreground color (in RRGGBBAA format)
    #[arg(long, default_value = "000000ff", value_parser = utils::str_to_color)]
    pub(crate) fg: Color,

    /// Background color (in RRGGBBAA format)
    #[arg(long, default_value = "ffff00ff", value_parser = utils::str_to_color)]
    pub(crate) bg: Color,
}
