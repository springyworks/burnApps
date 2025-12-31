# BachModem - Musical Wavelet Modem for HF Radio

A data modem for HF radio weak signal communication that uses Bach themes to communicate.
Transposes the WaveletsJAX Python/JAX implementation to Rust using Burn on GPU.

## Features

- **Morlet Wavelet Generation**: Gaussian-windowed complex exponentials on GPU
- **Musical Frequency Mapping**: 16-tone C-Major scale (261.63 - 1174.66 Hz)
- **FH-DPSK Modulation**: Frequency-Hopping Differential Phase Shift Keying
- **Melodic Hopping Pattern**: Pseudo-random musical interval jumps
- **Bach Preamble**: Fast arpeggio synchronization (C4-C6 sweep)
- **Musical Flourishes**: Periodic fast arpeggios throughout transmission (like Bach Preludes)
- **FFT-Based Synchronization**: O(N log N) correlation using CubeCL/Wgpu
- **Time-Slotted Repetition Protocol**: 15 repetitions with 5s listening gaps for -30 dB SNR
- **Deep-Space Performance**: Tested at -30 dB SNR over HF-Watterson channel

## Physical Layer Specification

- **Sampling Rate**: 8000 Hz
- **Symbol Duration**: 0.1 seconds (10 baud)
- **Carrier Frequencies**: C4 (261.63 Hz) to D6 (1174.66 Hz)
- **Wavelet Type**: Morlet (Gabor) wavelet
- **Modulation**: FH-DPSK with Lag-16 differential encoding
- **Bandwidth**: 200 Hz - 2.8 kHz (mono audio, SSB modulation)
- **Preamble**: 10x fast arpeggio sweep (~30 seconds)

## Usage

### Generate Transmission with Musical Flourishes

```bash
# Main demo - produces WAV file with Bach-themed transmission
cargo run --release --bin bachmodem
```

### Test Weak Signal Protocol (-30 dB)

```bash
# Validates the 15-repetition Time-Slotted Protocol
cargo run --release --example time_slot_test
```

### Test WAV Decoding

```bash
# Validates end-to-end WAV file processing
cargo run --release --example wav_decode_test
```

### Generate Clean Test Signal

```bash
# Generates a clean WAV with 5 repetitions, 5s gaps, and +5dB SNR
# Useful for listening tests and verifying the "Preamble-Data-Flourish" sequence
cargo run --release --example generate_clean_wav
```

- **Aesthetics**: Breaks up long transmissions with rapid upward arpeggios
- **Synchronization**: Provides periodic checkpoints for receiver re-sync
- **Channel Probing**: Sweeps all frequencies to measure fading

See [FLOURISHES.md](FLOURISHES.md) for detailed information.

**Example:**
```rust
use bachmodem::{modulate_fhdpsk_with_flourishes, write_wav};

let signal = modulate_fhdpsk_with_flourishes(
    &device,
    message.as_bytes(),
    true,  // Add preamble
    64,    // Insert flourish every 64 symbols (~2 minutes)
);
write_wav(&signal, "output.wav")?;
```

## References

Based on WaveletsJAX: `/home/rustuser/projects/pyth/WaveletsJAX`
- Achieves -30 dB SNR communication
- Polar codes for FEC (N=256, K=128)
- Musical modulation for aesthetic appeal
