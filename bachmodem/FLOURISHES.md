# Musical Flourishes in BachModem

## Overview

BachModem implements **Musical Flourishes** - periodic fast arpeggio sweeps (Bach Sweeps) inserted throughout data transmission. These serve both **aesthetic** and **functional** purposes, making the transmission sound like a Bach prelude while providing synchronization checkpoints.

## What are Flourishes?

A flourish is a **2-cycle Shifted Bach Sweep** - a rapid upward and downward arpeggio through all 16 tones in the C-Major scale, shifted by a musical interval (Dominant/Fifth).

**Musical characteristics:**
- Duration: 1.6 seconds (2 cycles)
- Pattern: Shifted Up Sweep (Shift 8) → Shifted Down Sweep (Shift 8)
- Speed: 20 notes/second (rapid but musically pleasing)
- Constant envelope: Unity amplitude throughout
- Processing gain: 54 dB correlation gain for detection

## Why Use Flourishes?

### 1. Musical Beauty
Without flourishes, the transmission consists of slow 0.1-second wavelet tones that gradually shift in frequency. While pleasant, this can become monotonous over long messages.

With flourishes every 32-64 symbols (~3-6 seconds), the transmission resembles **Bach's Preludes** with their characteristic fast arpeggio flourishes punctuating slower melodic passages.

### 2. Re-synchronization Checkpoints
If the receiver loses synchronization during transmission (due to deep fading, interference, or timing drift), flourishes provide:
- **Known waveform patterns** for cross-correlation
- **Shifted pattern** distinguishes them from Preamble and Post-amble
- **Periodic reset points** to recover sync without retransmission

### 3. Channel Probing
Each flourish sweeps through all 16 frequencies, providing real-time information about:
- Frequency-selective fading
- Doppler spread
- Available bandwidth
- Channel quality per frequency

## Post-amble

BachModem now includes a **Post-amble** to signal the end of transmission.
- **Pattern**: Shifted Up Sweep (Shift 4) → Shifted Down Sweep (Shift 4)
- **Duration**: 1.6 seconds
- **Function**: Explicit "End of Message" marker

## Usage

### Encoding with Flourishes

```rust
use bachmodem::{modulate_fhdpsk_with_flourishes, write_wav};
use burn::backend::Wgpu;

let device = Default::default();
let message = b"Hello, Bach!";

// Insert flourish every 64 symbols
let signal = modulate_fhdpsk_with_flourishes::<Wgpu>(
    &device,
    message,
    true,  // Add preamble and postamble
    64,    // Flourish interval (0 = disabled)
);

write_wav(&signal, "output.wav")?;
```

**Recommended intervals:**
- `0`: No flourishes (simplest, longest messages)
- `32`: Frequent (every ~3 seconds, most robust)
- `64`: Standard (every ~6 seconds, good balance)

### Decoding with Flourishes

The decoder **must know the flourish interval** used during encoding:

```rust
use bachmodem::{demodulate_fhdpsk_ex, read_wav};
use burn::backend::Wgpu;

let device = Default::default();
let signal = read_wav::<Wgpu>(&device, "output.wav")?;

// Same interval as encoding!
let decoded_bytes = demodulate_fhdpsk_ex::<Wgpu>(
    &device,
    &signal,
    true,  // Use preamble sync
    64,    // Must match encoding interval
);
```

⚠️ **Critical:** If the decoder uses the wrong flourish interval, it will:
- Extract flourishes as data symbols → decoding errors
- Skip data symbols as flourishes → lost data
- Misalign differential decoding → complete failure

## Implementation Details

### Encoder Behavior

```rust
for (i, &melody_idx) in melody_indices.iter().enumerate() {
    // Insert flourish at periodic intervals
    if flourish_interval > 0 && i > 0 && i % flourish_interval == 0 {
        let flourish = generate_bach_flourish::<B>(device);
        waveforms.push(flourish);
    }
    
    // Append data symbol
    let waveform = generate_symbol::<B>(device, melody_idx, phase, ...);
    waveforms.push(waveform);
}
```

**Timing:**
- Flourishes are inserted BEFORE the symbol at positions 64, 128, 192, ...
- No flourish before symbol 0 (preamble already provides initial sync)
- Each flourish adds 12,800 samples (1.6 seconds at 8 kHz)

### Decoder Behavior

```rust
let mut pos = 0;
let mut symbol_idx = 0;

while pos + symbol_len <= signal_len {
    // Skip flourish at expected positions
    if flourish_interval > 0 && symbol_idx > 0 && symbol_idx % flourish_interval == 0 {
        pos += flourish_len;  // Advance past flourish
        if pos + symbol_len > signal_len {
            break;
        }
    }
    
    // Extract data symbol
    let chunk = signal.slice([pos..pos + symbol_len]);
    symbol_chunks.push(chunk);
    pos += symbol_len;
    symbol_idx += 1;
}
```

**Synchronization:**
1. Find preamble via cross-correlation
2. Skip past preamble (3.2 seconds)
3. Extract symbols, skipping flourishes at known positions
4. Perform matched filtering and differential decoding

## Performance Impact

### Overhead

For a message with `N` data symbols and flourish interval `I`:
- Number of flourishes: `floor(N / I)`
- Flourish overhead: `floor(N / I) × 1.6 seconds`
- Relative overhead: `(floor(N / I) × 1.6) / (N × 0.1) = 16 / I`

Examples:
- `I = 32`: 50% overhead (High, but very robust)
- `I = 64`: 25% overhead (Standard)
- `I = 128`: 12.5% overhead (Low)

### Benefits

- **Synchronization recovery:** Can recover from sync loss without retransmission
- **Channel estimation:** Real-time SNR and fading information per frequency
- **Aesthetic value:** Much more pleasant to listen to during long transmissions

## Musical Considerations

### Timing Choices

The flourish interval should be musically pleasing:
- Too frequent (< 32): Feels repetitive and rushed
- Sweet spot (32-64): Natural breathing points in the "melody"
- Too sparse (> 128): Loses the Bach prelude character

### Integration with Melody

The FH-DPSK hop pattern creates a **melodic sequence** that varies with data content. Flourishes act as:
- **Cadences** - marking phrase boundaries
- **Recapitulations** - returning to the full scale
- **Punctuation** - separating "musical thoughts"

This creates a structure similar to Bach's keyboard preludes with their alternating arpeggiated sections and melodic passages.

## Testing

The `decoder_test.rs` example demonstrates perfect encoding/decoding with flourishes:

```bash
$ cargo run --release --example decoder_test

Original message: 174 bytes
Signal: 2972.0 seconds with 21 flourishes

Decoded: 174 bytes
Errors: 0 bytes
BER: 0.00%

✨ PERFECT DECODING! ✨
```

## Comparison with Other Modes

| Mode | Command | Use Case |
|------|---------|----------|
| No flourishes | `modulate_fhdpsk(..., 0)` | Shortest messages, maximum data rate |
| Sparse flourishes | `modulate_fhdpsk_with_flourishes(..., 128)` | Long messages, good channel |
| Standard flourishes | `modulate_fhdpsk_with_flourishes(..., 64)` | General use, balanced |
| Frequent flourishes | `modulate_fhdpsk_with_flourishes(..., 32)` | Poor channel, frequent fading |

## Future Enhancements

### Adaptive Flourishing
Automatically adjust flourish rate based on:
- Channel quality (more flourishes in poor conditions)
- Message priority (sparse for bulk data, frequent for critical)
- Available bandwidth

### Flourish Detection
Allow decoder to **detect** flourishes automatically instead of requiring the interval:
- Cross-correlate with Bach Sweep template
- Identify positions of high correlation peaks
- Infer flourish interval from spacing
- Extract data symbols between flourishes

### Variable Flourishes
Use different sweep patterns:
- Descending sweeps (cadential effect)
- Partial sweeps (3-5 tones, faster)
- Harmonic sweeps (octave jumps)
- Tempo variations (ritardando at end)

## References

- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical details of FH-DPSK
- [AESTHETICS.md](AESTHETICS.md) - Musical design philosophy
- [DETECTION_OPTIMIZATION.md](DETECTION_OPTIMIZATION.md) - Bach Sweep correlation gains
- [wavelet.rs](src/wavelet.rs) - `generate_bach_flourish()` implementation
- [modulation.rs](src/modulation.rs) - Encoder/decoder with flourish handling

---

*"The flourish is not mere ornament, but structural necessity - the breath between phrases, the return to tonic, the affirmation of key."*  
— Inspired by Bach's Preludes
