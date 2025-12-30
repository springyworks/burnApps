# Rust Burn FFT GPU App

This application demonstrates how to perform FFT on GPU using the Burn library.

## Prerequisites

- Rust installed
- GPU with Vulkan, Metal, or DX12 support (or CUDA if configured)
- Burn library (local dependency)

## Running

```bash
cargo run --release
```

## Structure

- `src/main.rs`: Entry point, sets up backend and data.
- `src/fft_kernel.rs`: Contains the FFT implementation logic (Radix-2 Cooley-Tukey).

## Status

- Basic structure implemented.
- Bit-reversal permutation implemented.
- Butterfly stages are currently placeholders.
