# webcam-demo

A demo written in Rust that utilizes the webcam ([`v4l`](https://github.com/raymanfx/libv4l-rs)), OpenGL ([`glow`](https://github.com/grovesNL/glow)) with textures and CUDA (using [`custos`](https://github.com/elftausend/custos) and [`nvjpeg-rs`](https://github.com/elftausend/nvjpeg-rs)) to do some real-time image processing with kernels.

Only runnable on linux systems (because of v4l).
Tested with a 30fps 1080p webcam (Logitech C920 PRO HD) and a RTX 2060 on Ubuntu 22.04.

## Dependencies
- Cuda Toolkit
    - nvrtc
    - nvjpeg

## Usage

Help:
```bash
cargo run --release -- -h
```
Run with default settings (sharpen filter):
```bash
cargo run --release
```
Run with different filters (sharpen, boxblur, overflow, marklight, edge, none):
```bash
cargo run --release -- -f boxblur
```

You may want to adjust the resolution of the video stream with the `-w` and `-h` flags. (default: 1920x1080)
```bash
cargo run --release -- -w 1280 -h 720
```

Sponsored by [www.geofront.eu](https://www.geofront.eu/) (internship)