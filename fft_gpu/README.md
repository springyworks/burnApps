# Real-time 2D FFT & Computer Vision with Rust Burn

A high-performance application demonstrating real-time 2D FFT, Sobel edge detection, and temporal difference analysis on camera input.

## Features
- **Dual Backend Support**:
  - **GPU**: Custom CubeCL kernels for 2D FFT and image processing.
  - **CPU**: Optimized pipeline using `burn-ndarray`, `rayon` (parallel processing), and `rustfft`.
- **Real-time Visualization**: Displays Input, FFT Magnitude, Sobel Edges, and Motion Energy side-by-side.
- **Camera Support**: Uses `nokhwa` for cross-platform video capture.

## Usage

### Run from Workspace Root
```bash
cargo run -p fft_gpu --release
```

### Run from this directory
```bash
cargo run --release
```

### Run on CPU (Optimized)
```bash
cargo run --release -- --on_cpu
```

### Generate Test Video
```bash
cargo run --release -- --generate-video > output.raw
```
