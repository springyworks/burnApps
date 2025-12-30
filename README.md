# Burn Applications Collection

This repository contains a collection of applications and experiments built using the [Burn](https://github.com/tracel-ai/burn) deep learning framework in Rust.

## Workspace Usage

This project is set up as a Cargo workspace. You can run any application from the root directory.

### List of Applications

- **[fft_gpu](./fft_gpu)**: Real-time 2D FFT & Computer Vision (GPU/CPU).

### Running Applications

To run an application, use `cargo run -p <app_name>`:

```bash
# Run the FFT GPU app
cargo run -p fft_gpu --release

# Run with arguments (e.g., CPU mode)
cargo run -p fft_gpu --release -- --on_cpu
```

## Applications Details

### [FFT GPU](./fft_gpu)
A high-performance application demonstrating real-time 2D FFT, Sobel edge detection, and temporal difference analysis.
- **Features**: Dual backend support (CubeCL for GPU, NdArray+Rayon for CPU), real-time camera visualization.
- **Tech Stack**: Burn, CubeCL, Nokhwa, Minifb.

---

*More applications will be added as they are developed.*
