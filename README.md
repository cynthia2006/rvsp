> [!WARNING]
> Development on this repository is *stalled* because of the complexities that arose during its development, and it is strongly advised to use [vsp](https://github.com/cynthia2006/vsp) instead (its C rewrite). Expect no updates, not even dependency updates.

# rvsp
rvsp is a lightweight audio spectrum visualizer made in Rust, drawing inspiration from [techniques used in WebAudio API][1]. It is GPU accelerated through OpenGL, supporting Linux and Raspberry PIs. 

> This uses PipeWire and Wayland/EGL (via **winit**/**glutin**), which is not available anywhere except Linux. Since my primary workstation is Linux, I have no plans to port it to Windows or Mac.

https://github.com/user-attachments/assets/a770dbe9-7315-40d6-a522-52dabebbcf86

## Installation

PipeWire development headers must be installed through the package manager of your distribution, if it packages them separately.

```sh
$ sudo apt install libpipewire-0.3-dev
$ cargo install --git "https://github.com/cynthia2006/rvsp" --locked
```

It installs a binary `rvsp` in `~/.cargo/bin`.

## Controls

- <kbd>↑</kbd> to increase, and <kbd>↓</kbd> to decrease gain.
- <kbd>←</kbd> to decrease, and <kbd>→</kbd> to increase smoothing time constant (0 < τ < 1).

### "Suckless" Philosophy

This app has no mechanism for dynamic configuration; if you want to tweak something, you can tweak it by **changing the constants** defined in `src/main.rs`. One might say, it satirizes the aforementioned (cult-like) philosophy.

Suppose you want to broaden the frequency range, you open `src/main.rs`; modify `MAX_FREQ` to, for example, `12000`.

```diff
-const MAX_FREQ: i32 = 10000;
+const MAX_FREQ: i32 = 12000;
```

Having saved the file, you may re-install the app (`--path` is the path where the repository is cloned).

```sh
$ cargo install --path . --locked
```

Or, you might execute `cargo run` to produce a debug build, and see if your changes fit; if not, re-edit, recompile, repeat.

### Smoothing time constant
Smoothing time constant (**τ**) is a parameter that controls how data is [smoothed exponentially](https://en.wikipedia.org/wiki/Exponential_smoothing) over time; a kind of lowpass filtering. In this context, it controls the responsiveness of the spectrum—higher the values, the less it is responsive. It defaults to `0.7`, which in my personal experience, provides the best experience.

## Spectrum

The **spectrum is linear**, both on the frequency and the amplitude axis. The frequency range is limited between **50 to 10000 Hz**; chiefly considering visual aesthetics, because usually one generally won't be interested in the higher parts of the spectrum.

To understand the theory that underpins the mechanics of this program, i.e. **STFT** (Short-Time Fourier Transform), read [this](https://brianmcfee.net/dstbook-site/content/ch09-stft/intro.html) article; and, for the choice of normalization, read [this](https://appliedacousticschalmers.github.io/scaling-of-the-dft/AES2020_eBrief/#31--scaling-of-dft-spectra-of-discrete-tones) (rather technical) article.

[1]: https://webaudio.github.io/web-audio-api/#fft-windowing-and-smoothing-over-time




