# Wavelet Mathematics in BachModem

## The Morlet (Gabor) Wavelet

### Definition

The Morlet wavelet is a Gaussian-windowed complex exponential:

$$\psi(t; f, s) = A \cdot e^{-\frac{t^2}{2s^2}} \cdot e^{i2\pi ft}$$

Where:
- $t$ is time (centered at 0 for each symbol)
- $f$ is the carrier frequency (one of 16 Bach scale frequencies)
- $s$ is the wavelet width parameter
- $A$ is the normalization constant for unit energy
- $i$ is the imaginary unit

### Components

1. **Gaussian Envelope**: $g(t) = e^{-\frac{t^2}{2s^2}}$
   - Provides time localization
   - Symmetric around $t = 0$
   - Contains 99.7% of energy within $\pm 3s$

2. **Complex Oscillation**: $o(t) = e^{i2\pi ft} = \cos(2\pi ft) + i\sin(2\pi ft)$
   - Provides frequency localization
   - Carrier frequency $f$
   - Euler's formula representation

### Energy Normalization

For unit energy ($\int_{-\infty}^{\infty} |\psi(t)|^2 dt = 1$):

$$|\psi(t)|^2 = A^2 \cdot e^{-\frac{t^2}{s^2}}$$

$$\int_{-\infty}^{\infty} A^2 e^{-\frac{t^2}{s^2}} dt = A^2 \cdot s\sqrt{\pi} = 1$$

$$A = \frac{1}{\sqrt{s\sqrt{\pi}}} = (s\sqrt{\pi})^{-1/2}$$

### Constant Width Design

In BachModem, we use:

$$s = \frac{T_s}{6}$$

Where $T_s = 2.0$ seconds is the symbol duration.

This ensures:
- All wavelets have the same time width regardless of frequency
- 6-sigma ($\pm 3s$) fits within the symbol window
- Constant energy across all frequencies
- Consistent time-frequency resolution

### Time Domain Implementation

For a symbol of duration $T_s$, sampled at $f_s = 8000$ Hz:

```rust
let num_samples = (T_s * f_s) as usize;
let s = T_s / 6.0;
let A = (s * sqrt(π))^(-0.5);

for i in 0..num_samples {
    let t = i / f_s - T_s / 2.0;  // Center at t=0
    let envelope = A * exp(-0.5 * (t/s)^2);
    let real_part = envelope * cos(2π * f * t);
    let imag_part = envelope * sin(2π * f * t);
}
```

## Phase Modulation

### Differential Phase Shift Keying (DPSK)

Each symbol carries a phase shift:

$$\psi_{\text{mod}}(t) = \psi(t) \cdot e^{i\phi}$$

Where $\phi \in \{0, \pi\}$ encodes the data bit:
- Bit 0 → $\phi = 0$ (no phase change)
- Bit 1 → $\phi = \pi$ (phase inversion)

The real transmitted signal is:

$$s(t) = \text{Re}\{\psi(t) \cdot e^{i\phi}\} = \text{Re}\{\psi(t)\}\cos\phi - \text{Im}\{\psi(t)\}\sin\phi$$

### Lag-16 Differential Encoding

For frequency-hopping resilience, we encode differentially with lag 16:

$$\phi[k, i] = \phi[k-1, i] + b[k, i] \cdot \pi$$

Where:
- $k$ is the block index (time)
- $i$ is the frequency index (0-15)
- $b[k, i]$ is the data bit at position $(k, i)$

This means each frequency maintains its own phase reference across hopping cycles.

## Matched Filter Reception

### Cross-Correlation

At the receiver, we correlate the received signal $r(t)$ with the complex conjugate reference:

$$R_i = \int_{-\infty}^{\infty} r(t) \cdot \psi_i^*(t) dt$$

Where $\psi_i^*(t)$ is the conjugate of the expected wavelet at frequency $f_i$.

### Phase Extraction

The correlation gives a complex number:

$$R_i = |R_i| e^{i\theta_i}$$

The phase $\theta_i = \arg(R_i)$ is extracted using:

$$\theta_i = \text{atan2}(\text{Im}(R_i), \text{Re}(R_i))$$

### Differential Decoding

For lag-16 decoding:

$$\Delta\theta_i[k] = \theta_i[k] - \theta_i[k-1]$$

Decision rule:
- If $|\Delta\theta_i| < \pi/2$ → bit = 0
- If $|\Delta\theta_i| > \pi/2$ → bit = 1

## Time-Frequency Resolution

### Heisenberg Uncertainty

The Morlet wavelet achieves the optimal time-frequency localization allowed by the Heisenberg uncertainty principle:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

For our wavelets:
- Time uncertainty: $\Delta t = s = T_s/6 = 0.333$ seconds
- Frequency uncertainty: $\Delta f = \frac{1}{2\pi s} = \frac{3}{\pi T_s} \approx 0.48$ Hz

This narrow frequency bandwidth makes the modem robust to adjacent channel interference.

## Processing Gain

### Symbol Integration

Each 2-second symbol contains $N = T_s \cdot f_s = 16,000$ samples.

Processing gain:

$$G_p = 10 \log_{10}(N) = 10 \log_{10}(16000) \approx 42 \text{ dB}$$

This explains how -30 dB SNR at the input becomes +12 dB at the symbol level after correlation.

### Energy Per Bit

With 1 bit per symbol at 2 seconds:

$$E_b/N_0 = \text{SNR}_{\text{input}} + G_p - 10\log_{10}(R)$$

Where $R = 0.5$ bps is the bit rate.

For -30 dB input SNR:

$$E_b/N_0 = -30 + 42 - 10\log_{10}(0.5) \approx 15 \text{ dB}$$

This is well above the Shannon limit for reliable communication!

## Frequency Hopping Pattern

### Musical Intervals

The hopping pattern `[0, 7, 4, 12, 2, 9, 5, 14, 1, 8, 3, 11, 6, 13, 10, 15]` creates musical intervals:

- 0 → 7: Perfect octave (C4 → C5)
- 7 → 4: Perfect fifth down (C5 → G4)
- 4 → 12: Perfect octave up (G4 → A5)
- etc.

This provides:
1. **Frequency diversity** against selective fading
2. **Pleasant melodic quality** for monitoring
3. **Pseudo-random appearance** to jammers
4. **Even spectral distribution** across the band

## Why Wavelets for HF Radio?

### Advantages Over Sinusoids

1. **Time Localization**: Wavelets have finite duration, allowing:
   - Clean symbol boundaries
   - Reduced inter-symbol interference
   - Better synchronization

2. **Multipath Resilience**: The Gaussian envelope naturally tapers off, reducing echoes

3. **Frequency Diversity**: Multiple carriers combat selective fading

4. **Processing Gain**: Long symbols enable deep integration for weak signal detection

### Comparison with Other Modulations

| Modulation | Time Loc. | Freq. Loc. | Complexity | SNR Sensitivity |
|------------|-----------|------------|------------|-----------------|
| FSK | Good | Poor | Low | -10 dB |
| PSK | Poor | Good | Medium | -5 dB |
| OFDM | Good | Good | High | 0 dB |
| **Wavelet** | **Optimal** | **Optimal** | Medium | **-30 dB** |

## Practical Considerations

### SSB Compatibility

The signal occupies 200 Hz - 2.8 kHz, fitting perfectly in a standard SSB passband (300 Hz - 3 kHz).

### Amplitude Distribution

The Gaussian envelope produces a near-constant amplitude signal (low PAPR), ideal for:
- Linear amplifiers
- Avoiding compression in SSB rigs
- Maximizing power efficiency

### Doppler Tolerance

The long 2-second symbols are relatively immune to Doppler shifts up to:

$$\Delta f_{\text{max}} = \frac{1}{2T_s} = 0.25 \text{ Hz}$$

This covers typical ionospheric movement at HF frequencies.

---

## References

1. Mallat, S. (2008). *A Wavelet Tour of Signal Processing*. Academic Press.
2. Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.
3. Proakis, J. G. (2001). *Digital Communications*. McGraw-Hill.
4. Sklar, B. (2001). *Digital Communications: Fundamentals and Applications*. Prentice Hall.

---

**Mathematical Beauty Meets Practical Engineering** 🎵📐📡
