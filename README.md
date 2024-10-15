# rvsp
rvsp is an audio spectrum visualizer in Rust, drawing inspiration from [techniques used in WebAudio API][1]. It uses GPU accelarated rendering (through OpenGL) for fast anti-aliased rendering. It is an unamibitious application, and intended to serve as more of a demo. If you require more advanced audio visualization software, you should checkout [ProjectM](https://github.com/projectM-visualizer/projectm).

## Controls

<kbd>k</kbd> to amplify spectrum, and <kbd>j</kbd> to attenuate spectrum.

[1]: https://webaudio.github.io/web-audio-api/#fft-windowing-and-smoothing-over-time
