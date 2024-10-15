use std::ops::{Div, Mul, Sub};

use skia_safe::Color;
use skia_safe::Paint;
use skia_safe::PaintStyle;
use skia_safe::Path;
use skia_safe::Point;
use skia_safe::Surface;

use realfft::num_complex::{Complex, ComplexFloat};

use crate::defs;

pub(crate) struct Renderer {
    /// A raster skia surface.
    surface: Surface,

    /// Polygonal path representing the spectrum.
    path: Path,

    /// Stroking style of the path.
    paint: Paint,

    /// Exponentially smoothed output.
    smoothed: Vec<f32>,

    points: Vec<Point>,

    /// Smoothing time consant (between 0 and 1). Higher value result in slower response times to changes in the
    /// frequency spectrum.
    ///
    /// https://webaudio.github.io/web-audio-api/#smoothing-over-time
    tau: f32,

    /// Minimum frequency to consider
    min_frequency: f32,

    /// Maximum frequency to consider
    max_frequency: f32,

    /// Clamping minimum (in decibels).
    db_min: f32,

    /// Clamping maximum (in decibels).
    db_max: f32,

    /// Background color.
    background: Color,

    /// Width of the window.
    width: f32,

    /// Height of the window.
    height: f32,
}

impl Renderer {
    pub(crate) fn new(
        surface: Surface,
        fft_size: usize,
        tau: f32,
        min_frequency: f32,
        max_frequency: f32,
        db_min: f32,
        db_max: f32,
        foreground: Color,
        background: Color,
        width: f32,
        height: f32,
    ) -> Self {
        let frequency_bin_count = fft_size / 2 + 1;
        let smoothed = vec![0f32; frequency_bin_count];

        let mut paint = Paint::default();

        paint
            .set_color(foreground)
            .set_stroke_width(1.2)
            .set_style(PaintStyle::Stroke)
            .set_anti_alias(true);

        Self {
            surface,
            path: Path::new(),
            paint,
            smoothed,
            points: vec![Point::new(width, 0.5 * height); frequency_bin_count],
            tau,
            min_frequency,
            max_frequency,
            db_min,
            db_max,
            background,
            width,
            height,
        }
    }

    pub(crate) fn render(&mut self, ft: &[Complex<f32>]) {
        let normalization = 2.0 / ft.len() as f32;
        let bin_width = 2048.0 / defs::SAMPLERATE as f32;

        let min_bin = (bin_width * self.min_frequency) as usize;
        let max_bin = (bin_width * self.max_frequency) as usize;
        let bin_delta = max_bin - min_bin;

        self.surface.canvas().draw_color(self.background, None);

        let mut x = 0.0;
        let mut sign = 1.0;

        let x_step = self.width as f32 / bin_delta as f32;

        for (i, bin) in ft[min_bin..max_bin].iter().enumerate() {
            // TODO Calculate needed space for smoothed output beforehand.
            let y = self.tau * self.smoothed[i] + (1.0 - self.tau) * bin.abs() * normalization;

            self.smoothed[i] = y;

            // FIXME Skia runs slow if the peaks are higher, so not using log-scale.
            // let y = y
            //         .log10().mul(20.0)
            //         .clamp(self.db_min, self.db_max)
            //         .sub(self.db_min)
            //         .div(self.db_max - self.db_min);

            self.points[i].set(x, 0.5 * self.height * (1.0 - sign * y));

            x += x_step;
            sign *= -1.0;
        }

        self.path
            .add_poly(&self.points[min_bin - 1..max_bin], false);
        self.surface.canvas().draw_path(&self.path, &self.paint);
        self.path.reset();
    }

    // TODO Re-add resizing ability
    // pub(crate) fn set_width(&mut self, width: f32) {
    //     self.width = width;
    // }

    // pub(crate) fn set_height(&mut self, height: f32) {
    //     self.height = height;
    // }
}
