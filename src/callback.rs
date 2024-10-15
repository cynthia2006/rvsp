use sdl2::audio::AudioCallback;

pub(crate) struct Callback {
    /// A circular buffer to hold incoming samples.
    pub buffer: Vec<f32>,
}

impl AudioCallback for Callback {
    type Channel = f32;

    fn callback(&mut self, samples: &mut [Self::Channel]) {
        if samples.len() > self.buffer.len() {
            self.buffer.clear();
            self.buffer
                .extend(samples.iter().nth(samples.len() - self.buffer.len()));
        } else {
            self.buffer.drain(0..samples.len());
            self.buffer.extend(samples.iter());
        }
    }
}
