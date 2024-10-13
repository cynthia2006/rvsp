use std::ops::{Mul, Sub, Div};

use realfft::num_complex::{Complex, ComplexFloat};

use sdl2::render::Canvas;
use sdl2::sys::{SDL_FRect, SDL_RenderFillRectsF};
use sdl2::pixels::Color;
use sdl2::video::Window;

use crate::utils;

pub(crate) struct Renderer {
    /// Handle to canvas.
    canvas: Canvas<Window>,
    
    /// Buffer of rectangles.
    rects: Vec<SDL_FRect>,

    /// Exponentially smoothed output.
    smoothed: Vec<f32>,

    /// Smoothing time consant (between 0 and 1). Higher value result in slower response times to changes in the
    /// frequency spectrum.
    /// 
    /// https://webaudio.github.io/web-audio-api/#smoothing-over-time
    tau: f32,

    /// Clamping minimum (in decibels).
    db_min: f32,

    /// Clamping maximum (in decibels).
    db_max: f32,

    /// Foreground color.
    foreground: Color,

    /// Background color.
    background: Color,

    /// Width of the window.
    width: f32,

    /// Height of the window.
    height: f32,
}

impl Renderer {
    pub(crate) fn new(
        canvas: Canvas<Window>,
        fft_size: usize,
        tau: f32,
        db_min: f32,
        db_max: f32,
        foreground: Color,
        background: Color,
        width: f32,
        height: f32
    ) -> Self {
        let frequency_bin_count = fft_size / 2 + 1;
        let rects = vec![utils::frect_new(); frequency_bin_count];
        let smoothed = vec![0f32; frequency_bin_count];

        Self {
            canvas,
            rects,
            smoothed,
            tau,
            db_min,
            db_max,
            foreground,
            background,
            width,
            height
        }
    }

    pub(crate) fn render(&mut self, ft: &[Complex<f32>]) {
        let x_step = self.width / ft.len() as f32;
        let normalization = 2.0 / ft.len() as f32;

        let mut x = 0.0;

        self.canvas.set_draw_color(self.background);
        self.canvas.clear();

        for (i, bin) in ft.iter().enumerate() {
            let y = self.tau * self.smoothed[i] + (1.0 - self.tau) * bin.abs() * normalization;
            
            self.smoothed[i] = y;

            let y = y
                .log10().mul(20.0)
                .clamp(self.db_min, self.db_max)
                .sub(self.db_min)
                .div(self.db_max - self.db_min);

            self.rects[i] = SDL_FRect {
                x,
                y: self.height * (1.0 - y),
                w: x_step,
                h: y * self.height,
            };

            x += x_step;
        }

        self.canvas.set_draw_color(self.foreground);

        // Canvas API does not support floating point coordinates but SDL itself does, and it is perfectly fine to take
        // this leap of faith as this line would be called anyway if the functionality was implemented in the library.
        unsafe {
            SDL_RenderFillRectsF(self.canvas.raw(), self.rects.as_ptr(), self.rects.len() as i32);
        }

        self.canvas.present();
    }
    
    pub(crate) fn set_width(&mut self, width: f32) {
        self.width = width;
    }
    
    pub(crate) fn set_height(&mut self, height: f32) {
        self.height = height;
    }
}
