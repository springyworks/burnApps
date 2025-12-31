# Detection Optimization Research Notes

## From WaveletsJAX Research: What Makes Signals Detectable?

Based on experimental validation at -30 to -34 dB SNR, here are the key factors:

## 1. **Constant Envelope is King**

### Why It Matters
- At -30 dB, the signal is 1/1000th the power of noise
- Any amplitude variation wastes precious signal energy
- PAPR (Peak-to-Average Power Ratio) directly reduces effective SNR

### The Proof
From bach_preamble_evaluation.md:
- **Polyphonic chords FAILED** at -30 dB
- **Reason**: High PAPR forced amplitude scaling, losing 7-10 dB
- **Single-tone sweep SUCCEEDED** at -32 dB
- **Reason**: Constant envelope maintained maximum power

### Implementation in BachModem
✅ Our Morlet wavelets have constant envelope (Gaussian * sinusoid)
✅ Single frequency per symbol (FH-DPSK)
✅ No chord/polyphony (would increase PAPR)

## 2. **Matched Filter Processing Gain**

### The Math
```
Processing Gain (dB) = 10 * log₁₀(N_samples)
For 2.0s at 8kHz: PG = 10 * log₁₀(16000) ≈ 42 dB
```

### What This Means
- Input: -30 dB SNR (buried in noise)
- After matched filtering: +12 dB SNR (clearly detectable)
- This is why long symbols work!

### Symbol Duration Trade-offs
| Duration | Samples | Proc. Gain | Data Rate | Detection |
|----------|---------|------------|-----------|-----------|
| 0.1s | 800 | 29 dB | 10 bps | Poor at -30 dB |
| 0.5s | 4000 | 36 dB | 2 bps | Marginal |
| 2.0s | 16000 | 42 dB | 0.5 bps | **Excellent** |
| 10.0s | 80000 | 49 dB | 0.1 bps | Overkill |

**Conclusion**: 2.0s is optimal sweet spot for -30 dB.

## 3. **Differential Encoding Resilience**

### Why DPSK > PSK at -30 dB

**Problem with absolute PSK:**
- Requires perfect frequency reference
- 1 Hz drift rotates phase by 360° in 1 second
- At -30 dB, carrier tracking is impossible

**Solution with DPSK:**
- Only measure phase *difference* between symbols
- Frequency offset cancels out (it's common to both)
- Lag-16 encoding: compare symbols at same frequency

### The Math
```
Symbol [k, freq_i] compared to Symbol [k-1, freq_i]
Even with 5 Hz drift, the Δphase is accurate!
```

## 4. **Frequency Diversity via Hopping**

### Why Musical Intervals Help

**Selective Fading Problem:**
- HF ionosphere creates nulls at specific frequencies
- A single carrier can be completely attenuated
- Duration: seconds to minutes

**Frequency Hopping Solution:**
- 16 different frequencies (261-1174 Hz span)
- Even if 3-4 frequencies are nulled, others work
- Melodic pattern ensures even spectral distribution

### Coverage Analysis
```
Bandwidth: 913 Hz (D6 - C4)
Spacing: ~60 Hz average between tones
Coherence bandwidth of HF: ~200 Hz typical
→ At least 4-5 independent fading channels
```

## 5. **Synchronization: The Bach Sweep Discovery**

### What Was Tested (from research)

| Method | Duration | PAPR | Result | Reason |
|--------|----------|------|--------|--------|
| Slow Arpeggio | Long | Low | ❌ Failed | Poor time resolution |
| Chords | Long | High | ❌ Failed | Power scaling loss |
| **Fast Sweep** | Long | Low | ✅ Success | **Best of both** |

### The Winner: Fast Arpeggio (Bach Sweep)
- **Constant envelope**: Each note at full power
- **Wide bandwidth**: 16 tones swept rapidly
- **Long duration**: 30 seconds for 54 dB gain
- **Sharp correlation peak**: Fast changes = good time resolution

### Why It Sounds Beautiful
The fast sweep (0.1s per note) sounds like:
- Bach's harpsichord preludes
- A music box winding up
- A bird's trill

But mathematically it's a **wideband constant-envelope synchronization signal**.

## 6. **Soft-Decision Decoding (Future FEC)**

### What Your Research Recommends

**Hard Decision (Bad):**
```
Correlation > threshold → Bit = 1
Correlation < threshold → Bit = 0
```
Throws away information about *how confident* you are.

**Soft Decision (Good):**
```
Correlation = 0.9 → Very confident it's a 1 → LLR = +3.5
Correlation = 0.1 → Very confident it's a 0 → LLR = -3.5
Correlation = 0.51 → Barely think it's 1 → LLR = +0.2
```

### Why This Matters at -30 dB
- At -30 dB, many symbols are "iffy" (correlation ~0.6)
- Polar codes can use these "soft" confidences
- With FEC: 0.6 × 0.55 × 0.7 × 0.65 → Strong signal after combining

**Your Python implementation achieves:**
- -34 dB SNR
- 0% BER
- With Polar codes (N=256, K=128)

## 7. **Watterson Channel Model (Future Testing)**

### What Makes HF Hard
Not just noise, but:
- **Multipath**: Signal arrives via multiple paths
  - Direct (F2 layer): 2000 km, 7 ms delay
  - Indirect (F1+F2): 2000 km, 9 ms delay
  - Creates 2 ms "echo" → phase interference

- **Doppler Spread**: Ionosphere moving
  - Typical: ±0.5 Hz spread
  - Creates time-varying phase rotation

- **Fading Rate**: Signal strength varies
  - Slow fading: 0.1 Hz (10s period)
  - Fast fading: 1 Hz (1s period)

### Why 2-Second Symbols Help
- Longer than typical fading period
- Integration averages out fast fluctuations
- Slow fading can be tracked via pilot tones

## Recommendations for BachModem Enhancement

### Priority 1: Detection Optimization
✅ Already implemented:
- Constant envelope wavelets
- 2-second symbols (42 dB gain)
- Differential encoding (Lag-16)
- Frequency hopping (16 tones)
- Bach Sweep preamble

### Priority 2: Add Soft-Decision Capability
Modify demodulation to output LLRs instead of hard bits:
```rust
pub fn demodulate_soft<B: Backend>(
    signal: &Tensor<B, 1>,
) -> Vec<f32> {  // LLRs instead of Vec<u8>
    // Return correlation magnitudes
    // Not just argmax(correlations)
}
```

### Priority 3: Implement Polar Code FEC
From your Python:
```python
N = 256  # Code length
K = 128  # Information bits
# Achieves -34 dB SNR!
```

Rust implementation needed:
- Polar encoder
- Successive Cancellation List (SCL) decoder
- CRC-8 for error detection

### Priority 4: Channel Simulation
Add Watterson channel model for testing:
- Multipath delays
- Doppler spread
- Rayleigh fading

## The "Orchestra" Concept

Your question: "what integrate wavelet orchestra if you like"

This is brilliant! Think of it as:

### Current (Single Instrument)
Each symbol = one Morlet wavelet = one flute note

### Future (Orchestra)
Multiple wavelets simultaneously:
- **Melody line** (data channel): Hopping frequencies
- **Bass line** (pilot channel): Fixed frequency for phase reference
- **Harmony** (FEC redundancy): Additional wavelets at perfect fifth/octave

**But**: Must maintain constant envelope!
- Use **OFDM-like** approach with clipping protection
- Or **time-multiplex** instead of frequency-multiplex

## Detection Quality Metrics

From your benchmarks:

| SNR (dB) | Sync Success | BER (no FEC) | BER (with Polar) |
|----------|--------------|--------------|------------------|
| -30 | 100% | ~15% | 0% |
| -32 | 100% | ~25% | 0% |
| -33 | 100% | ~35% | 0% |
| -34 | 100% | ~45% | 0% |
| -35 | 95% | ~55% | 2% |

**Key insight**: Synchronization works even when raw BER is 45%!
This is because:
- Preamble gets 54 dB processing gain (30s integration)
- Data gets 42 dB processing gain (2s integration)
- 12 dB difference explains the threshold

## Final Answer to Your Question

**"What wavelet scheme is best for decoding?"**

### The Winner: Morlet with These Optimizations

1. **Constant envelope** (not chirps, not variable amplitude)
2. **Long integration** (2s minimum for -30 dB)
3. **Differential encoding** (defeats frequency drift)
4. **Frequency diversity** (hopping combats selective fading)
5. **Soft-decision FEC** (Polar codes for the win)

Your Python research proved this at **-34 dB SNR, 0% BER**.

The Rust/Burn implementation now has all the core pieces.
Next step: Add the Polar code FEC layer!

---

**The "Orchestra" insight is correct: multiple wavelets can help, but only if you maintain constant envelope via time-multiplexing or careful OFDM design.**
