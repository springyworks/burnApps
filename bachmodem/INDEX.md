# BachModem Complete Documentation Index

Welcome to **BachModem** - Where Bach's Music Meets HF Radio Communications! 🎵📡

## Quick Navigation

### 📘 Getting Started
1. **[README.md](README.md)** - Start here! Quick overview and usage
2. **[examples/simple.rs](examples/simple.rs)** - Simple code example

### 🔧 Technical Documentation
3. **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Implementation details and feature comparison
4. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture diagrams and data flow
5. **[MATHEMATICS.md](MATHEMATICS.md)** - Wavelet theory, signal processing math

### 🎨 Philosophy & Vision
6. **[AESTHETICS.md](AESTHETICS.md)** - The art and philosophy of musical modems
7. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete project overview

## Document Purpose Guide

### If you want to...

**...get started quickly:**
→ [README.md](README.md) + [examples/simple.rs](examples/simple.rs)

**...understand the implementation:**
→ [IMPLEMENTATION.md](IMPLEMENTATION.md) + [ARCHITECTURE.md](ARCHITECTURE.md)

**...learn the mathematics:**
→ [MATHEMATICS.md](MATHEMATICS.md)

**...appreciate the artistic vision:**
→ [AESTHETICS.md](AESTHETICS.md)

**...see the big picture:**
→ [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

## Source Code Guide

### Core Modules

| File | Lines | Purpose |
|------|-------|---------|
| [src/lib.rs](src/lib.rs) | 16 | Public library interface |
| [src/main.rs](src/main.rs) | 108 | Demo application |
| [src/wavelet.rs](src/wavelet.rs) | 178 | Morlet wavelet generation |
| [src/modulation.rs](src/modulation.rs) | 243 | FH-DPSK modulation/demodulation |
| [src/wav.rs](src/wav.rs) | 114 | WAV file I/O |

**Total:** ~700 lines of Rust

### Examples

| File | Purpose |
|------|---------|
| [examples/simple.rs](examples/simple.rs) | Basic usage demonstration |

## Key Concepts by Document

### README.md
- Quick start guide
- Installation
- Basic usage
- Feature overview

### IMPLEMENTATION.md
- Feature checklist
- Python vs Rust comparison
- Technical specifications
- What's implemented vs future work

### ARCHITECTURE.md
- Signal flow diagrams
- Module structure
- Wavelet visualization
- Frequency hopping pattern
- Phase encoding
- Processing gain breakdown

### MATHEMATICS.md
- Morlet wavelet equations
- Energy normalization
- Time-frequency localization
- Heisenberg uncertainty
- Phase modulation theory
- Matched filter reception
- Processing gain calculation

### AESTHETICS.md
- Philosophy of musical modems
- Why Bach?
- The sound and aesthetics
- Deep space communications
- Cultural references
- Listening guide

### PROJECT_SUMMARY.md
- Complete feature list
- Technical achievements
- Usage instructions
- Performance metrics
- Future roadmap

## Quick Reference Cards

### Physical Layer
```
Sampling Rate:     8000 Hz
Symbol Duration:   0.1 seconds (10 baud)
Modulation:        FH-DPSK
Frequencies:       261.63 - 1174.66 Hz (C-Major)
Bandwidth:         200 Hz - 2.8 kHz
Target SNR:        -30 dB
Bit Rate:          ~10 bps (raw)
Processing Gain:   ~29 dB
```

### Bach Scale
```
C4=261.63  D4=293.66  E4=329.63  F4=349.23
G4=392.00  A4=440.00  B4=493.88  C5=523.25
D5=587.33  E5=659.25  F5=698.46  G5=783.99
A5=880.00  B5=987.77  C6=1046.50 D6=1174.66
```

### Hopping Pattern
```
0→7→4→12→2→9→5→14→1→8→3→11→6→13→10→15
(Creates melodic intervals)
```

## Code Examples

### Basic Usage
```rust
use bachmodem::{modulate_fhdpsk, write_wav};
use burn::backend::Wgpu;

let device = Default::default();
let message = b"Hello, World!";
let signal = modulate_fhdpsk::<Wgpu>(&device, message, true);
write_wav(&signal, "output.wav")?;
```

### Custom Wavelet Generation
```rust
use bachmodem::wavelet::{morlet_wavelet, BACH_FREQUENCIES};

let device = Default::default();
let frequency = BACH_FREQUENCIES[5]; // A4 = 440 Hz
let (real, imag) = morlet_wavelet::<Wgpu>(
    &device, 
    frequency, 
    0.1,    // duration
    8000.0  // sample rate
);
```

## Visual Guides

### Signal Flow
```
Message → Bits → Differential Phase → 
Wavelet Generation → Preamble → WAV File
```

### Module Dependencies
```
main.rs
  ↓
lib.rs ─┬─→ wavelet.rs
        ├─→ modulation.rs
        ├─→ fft_correlation.rs
        └─→ wav.rs
```

## Learning Path

### Beginner
1. Read [README.md](README.md)
2. Run the demo: `cargo run --release`
3. Listen to `bachmodem_output.wav`
4. Try [examples/simple.rs](examples/simple.rs)

### Intermediate
5. Study [IMPLEMENTATION.md](IMPLEMENTATION.md)
6. Read [src/wavelet.rs](src/wavelet.rs) code
7. Understand [ARCHITECTURE.md](ARCHITECTURE.md)
8. Experiment with different messages

### Advanced
9. Deep dive into [MATHEMATICS.md](MATHEMATICS.md)
10. Implement receiver (demodulation)
11. Add FEC (Polar codes)
12. Integrate with SDR hardware

### Philosopher
13. Read [AESTHETICS.md](AESTHETICS.md)
14. Contemplate the nature of communication
15. Create art with modulated signals
16. Share beautiful data transmissions

## External Resources

### Related Projects
- **WaveletsJAX**: Original Python/JAX implementation
  - Location: `/home/rustuser/projects/pyth/WaveletsJAX`
  - Includes: Polar codes FEC, full receiver, benchmarks

### Frameworks
- **Burn**: https://burn.dev
- **WGPU**: https://wgpu.rs
- **Hound**: https://crates.io/crates/hound

### Background Reading
- Mallat - "A Wavelet Tour of Signal Processing"
- Proakis - "Digital Communications"
- ARRL Handbook - Amateur Radio techniques

## Support & Contribution

### Questions?
- Check documentation first
- Review examples
- Inspect source code comments

### Want to Contribute?
- Implement receiver/demodulation
- Add FEC (Polar codes)
- Create more examples
- Improve documentation
- Add tests

### Share Your Creations!
- Generate interesting messages
- Create musical compositions
- Test over real HF links
- Make art with the signals

## FAQ Quick Links

**Q: How do I change the message?**
→ Edit [src/main.rs](src/main.rs) line 27

**Q: How does the wavelet work?**
→ See [MATHEMATICS.md](MATHEMATICS.md) section "The Morlet Wavelet"

**Q: Why is it so slow (0.1 seconds per symbol)?**
→ See [AESTHETICS.md](AESTHETICS.md) section "Why 2-Second Symbols?" (Historical note: we sped it up!)

**Q: Can I use this for actual communication?**
→ Yes! But add receiver implementation first (future work)

**Q: Why Bach?**
→ See [AESTHETICS.md](AESTHETICS.md) section "Why Bach?"

**Q: What's the data rate?**
→ ~10 bits/second (see [IMPLEMENTATION.md](IMPLEMENTATION.md))

**Q: How does it work at -30 dB?**
→ See [MATHEMATICS.md](MATHEMATICS.md) section "Processing Gain"

## Version History

**v0.2.0** (2025-01-01)
- FFT-based Synchronization (O(N log N))
- Time-Slotted Repetition Protocol
- Full GPU acceleration (CubeCL/Wgpu)
- 10 baud operation

**v0.1.0** (2025-12-31)
- Initial release
- Core wavelet generation
- FH-DPSK modulation
- Bach preamble
- WAV output
- Comprehensive documentation

## License

(To be determined by project owner)

## Credits

- **Original Research**: User's WaveletsJAX implementation
- **Framework**: Burn.rs team
- **Inspiration**: Johann Sebastian Bach, amateur radio community
- **Philosophy**: "Let data be beautiful"

---

## Navigation Tips

### By Topic

**Wavelets**: [MATHEMATICS.md](MATHEMATICS.md) + [src/wavelet.rs](src/wavelet.rs)  
**Modulation**: [IMPLEMENTATION.md](IMPLEMENTATION.md) + [src/modulation.rs](src/modulation.rs)  
**Philosophy**: [AESTHETICS.md](AESTHETICS.md)  
**Code**: [src/](src/) directory  
**Examples**: [examples/](examples/) directory  

### By Skill Level

**Casual User**: [README.md](README.md)  
**Developer**: [IMPLEMENTATION.md](IMPLEMENTATION.md) + [ARCHITECTURE.md](ARCHITECTURE.md)  
**Researcher**: [MATHEMATICS.md](MATHEMATICS.md)  
**Artist**: [AESTHETICS.md](AESTHETICS.md)  

---

**Welcome to BachModem!**  
**Where every transmission is a symphony.** 🎵

**73 de BachModem** 📡✨
