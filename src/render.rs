use skia_safe::Color;
use skia_safe::Paint;
use skia_safe::PaintStyle;
use skia_safe::Path;
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

    /// Smoothing time consant (between 0 and 1). Higher value result in slower response times to changes in the
    /// frequency spectrum.
    ///
    /// https://webaudio.github.io/web-audio-api/#smoothing-over-time
    tau: f32,

    /// Background color.
    background: Color,

    // Miscellaneous state info.
    highest_bin: usize,
    lowest_bin: usize,
}

impl Renderer {
    pub(crate) fn new(
        surface: Surface,
        tau: f32,
        min_frequency: f32,
        max_frequency: f32,
        foreground: Color,
        background: Color
    ) -> Self {
        let bin_width = defs::FFT_SIZE as f32 / defs::SAMPLERATE as f32;
        let lowest_bin = (bin_width * min_frequency) as usize;
        let highest_bin = (bin_width * max_frequency) as usize;

        let smoothed = vec![0f32; highest_bin - lowest_bin];

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
            tau,
            background,
            lowest_bin,
            highest_bin,
        }
    }

    pub(crate) fn render(&mut self, ft: &[Complex<f32>]) {
        const NORMALIZATION: f32 = 2.0 / defs::FFT_SIZE as f32;

        self.surface.canvas().draw_color(self.background, None);

        let width = self.surface.width() as f32;
        let height = self.surface.height() as f32;

        let x_step = width / (self.highest_bin - self.lowest_bin) as f32;
        let mut x = x_step;
        let mut sign = 1.0;

        self.path.move_to((0.0, 0.5 * height));

        for (i, bin) in ft[self.lowest_bin..self.highest_bin].iter().enumerate() {
            let y = self.tau * self.smoothed[i] + (1.0 - self.tau) * bin.abs() * NORMALIZATION;
            self.smoothed[i] = y;

            self.path.line_to((x, 0.5 * height * (1.0 - sign * y)));

            x += x_step;
            sign *= -1.0;
        }

        self.surface.canvas().draw_path(&self.path, &self.paint);
        self.path.reset();
    }

    pub(crate) fn set_surface(&mut self, surface: Surface) {
        self.surface = surface;
    }
}
