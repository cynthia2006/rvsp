# rvsp
rvsp is an audio spectrum visualizer in Rust, drawing inspiration from [techniques used in WebAudio API][1]. It is GPU accelarated with 8x MSAA. It is more of a hobby project, intended to serve as sort of a demo; for serious needs, checkout [ProjectM](https://github.com/projectM-visualizer/projectm).

> This program only runs on Linux. It uses PipeWire for capture, which is not available for any other platform; also explicitly uses the Wayland/EGL stack through **winit** and **glutin**.

## Controls

- <kbd>↑</kbd> to increase, and <kbd>↓</kbd> to decrease gain.
- <kbd>←</kbd> to decrease, and <kbd>→</kbd> to increase smoothing time constant (0 < τ < 1).

### Smoothing time constant
Smoothing time constant (τ) is a parameter that controls how data is [smoothed exponentially](https://en.wikipedia.org/wiki/Exponential_smoothing) over time; a kind of lowpass filtering. In this context, it controls the responsivity of the spectrum—higher the values, the less it is responsive.

[1]: https://webaudio.github.io/web-audio-api/#fft-windowing-and-smoothing-over-time

## The Demo


https://github.com/user-attachments/assets/73e62204-8bf3-47b6-acdb-f7581fd14a24

