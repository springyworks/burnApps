use burn::tensor::{Tensor, backend::Backend};
use std::f64::consts::PI;

/// Bach Scale Frequencies (C-Major, C4 to D6)
pub const BACH_FREQUENCIES: [f64; 16] = [
    261.63,  // C4 - 0x0
    293.66,  // D4 - 0x1
    329.63,  // E4 - 0x2
    349.23,  // F4 - 0x3
    392.00,  // G4 - 0x4
    440.00,  // A4 - 0x5
    493.88,  // B4 - 0x6
    523.25,  // C5 - 0x7
    587.33,  // D5 - 0x8
    659.25,  // E5 - 0x9
    698.46,  // F5 - 0xA
    783.99,  // G5 - 0xB
    880.00,  // A5 - 0xC
    987.77,  // B5 - 0xD
    1046.50, // C6 - 0xE
    1174.66, // D6 - 0xF
];

/// Frequency-Hopping Pattern (Melodic Intervals)
/// Creates pleasant musical jumps instead of linear progression
pub const HOPPING_PATTERN: [usize; 16] = [0, 7, 4, 12, 2, 9, 5, 14, 1, 8, 3, 11, 6, 13, 10, 15];

/// Physical Layer Parameters
pub const FS: f64 = 8000.0;              // Sampling frequency (Hz)
pub const SYMBOL_DURATION: f64 = 0.1;    // Symbol duration (seconds) - Fast for testing (spec: 2.0s for deep space)
pub const PREAMBLE_NOTE_DURATION: f64 = 0.05; // Preamble note duration (seconds)

/// Generates the melody hopping sequence for a given number of symbols
pub fn get_melody_indices(num_symbols: usize) -> Vec<usize> {
    (0..num_symbols)
        .map(|i| HOPPING_PATTERN[i % HOPPING_PATTERN.len()])
        .collect()
}

/// Generates a Morlet (Gabor) wavelet
/// 
/// ψ(t; f, s) = A · exp(-t²/2s²) · exp(i·2πf·t)
/// 
/// Where:
/// - A = (s√π)^(-1/2) [Normalization for unit energy]
/// - s = duration / 6 [Wavelet width parameter, 6-sigma fits in window]
/// - f = carrier frequency
/// - t ∈ [-duration/2, duration/2]
pub fn morlet_wavelet<B: Backend>(
    device: &B::Device,
    frequency: f64,
    duration: f64,
    fs: f64,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let num_samples = (duration * fs) as usize;
    let s = duration / 6.0; // Wavelet width parameter
    
    // Time vector: [-duration/2, duration/2]
    let t_values: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = (i as f64) / fs - duration / 2.0;
            t as f32
        })
        .collect();
    
    let t = Tensor::<B, 1>::from_floats(t_values.as_slice(), device);
    
    // Gaussian envelope: A * exp(-0.5 * (t/s)²)
    // A = (s * sqrt(π))^(-0.5)
    let norm_factor = (s * PI.sqrt()).powf(-0.5);
    let envelope = t.clone()
        .powf_scalar(2.0)
        .div_scalar((2.0 * s * s) as f32)
        .neg()
        .exp()
        .mul_scalar(norm_factor as f32);
    
    // Oscillatory components
    // Real: cos(2πft)
    // Imag: sin(2πft)
    let omega = 2.0 * PI * frequency;
    let phase = t.mul_scalar(omega as f32);
    
    let real_part = phase.clone().cos().mul(envelope.clone());
    let imag_part = phase.sin().mul(envelope);
    
    (real_part, imag_part)
}

/// Generates a single symbol waveform (real part only for transmission)
pub fn generate_symbol<B: Backend>(
    device: &B::Device,
    symbol_idx: usize,
    phase_offset: f64,
    duration: f64,
    fs: f64,
) -> Tensor<B, 1> {
    let frequency = BACH_FREQUENCIES[symbol_idx];
    let (real, imag) = morlet_wavelet::<B>(device, frequency, duration, fs);
    
    // Apply phase shift: wavelet * exp(i * phase_offset)
    // Real part: real * cos(phase) - imag * sin(phase)
    let cos_phase = phase_offset.cos() as f32;
    let sin_phase = phase_offset.sin() as f32;
    
    real.mul_scalar(cos_phase).sub(imag.mul_scalar(sin_phase))
}

/// Generates the Bach Preamble (Fast Arpeggio Sweep)
/// 
/// Sweeps UP-DOWN-UP-DOWN (4 cycles).
/// Standard C-Major scale (Shift 0).
pub fn generate_bach_preamble<B: Backend>(device: &B::Device) -> Tensor<B, 1> {
    let note_duration = PREAMBLE_NOTE_DURATION;
    let mut sequence = Vec::new();
    
    // Cycle 1: Up (Shift 0)
    sequence.extend(get_shifted_sweep_up(0));
    // Cycle 2: Down (Shift 0)
    sequence.extend(get_shifted_sweep_down(0));
    // Cycle 3: Up (Shift 0)
    sequence.extend(get_shifted_sweep_up(0));
    // Cycle 4: Down (Shift 0)
    sequence.extend(get_shifted_sweep_down(0));
    
    generate_from_sequence::<B>(device, &sequence, note_duration)
}

/// Generates Bach Flourish / Inter-amble
/// 
/// Shifted UP-DOWN sweep (Shift 8 - Dominant/Fifth).
/// Distinct from Preamble but musically related.
pub fn generate_bach_flourish<B: Backend>(device: &B::Device) -> Tensor<B, 1> {
    let note_duration = PREAMBLE_NOTE_DURATION;
    let mut sequence = Vec::new();
    
    // Up (Shift 8)
    sequence.extend(get_shifted_sweep_up(8));
    // Down (Shift 8)
    sequence.extend(get_shifted_sweep_down(8));
    
    generate_from_sequence::<B>(device, &sequence, note_duration)
}

/// Generates Bach Post-amble
/// 
/// Shifted UP-DOWN sweep (Shift 4 - Mediant/Third).
/// Signals end of transmission.
pub fn generate_bach_postamble<B: Backend>(device: &B::Device) -> Tensor<B, 1> {
    let note_duration = PREAMBLE_NOTE_DURATION;
    let mut sequence = Vec::new();
    
    // Up (Shift 4)
    sequence.extend(get_shifted_sweep_up(4));
    // Down (Shift 4)
    sequence.extend(get_shifted_sweep_down(4));
    
    generate_from_sequence::<B>(device, &sequence, note_duration)
}

/// Helper: Get indices for a shifted UP sweep
fn get_shifted_sweep_up(shift: usize) -> Vec<usize> {
    (0..16).map(|i| (i + shift) % 16).collect()
}

/// Helper: Get indices for a shifted DOWN sweep
fn get_shifted_sweep_down(shift: usize) -> Vec<usize> {
    (0..16).rev().map(|i| (i + shift) % 16).collect()
}

/// Generates Bach Sweep UP (Legacy helper, kept for compatibility if needed)
fn generate_bach_sweep_up<B: Backend>(device: &B::Device, cycles: usize) -> Tensor<B, 1> {
    let note_duration = PREAMBLE_NOTE_DURATION;
    let mut sequence = Vec::new();
    for _ in 0..cycles {
        sequence.extend(get_shifted_sweep_up(0));
    }
    generate_from_sequence::<B>(device, &sequence, note_duration)
}

/// Generates Bach Sweep DOWN (Legacy helper)
fn generate_bach_sweep_down<B: Backend>(device: &B::Device, cycles: usize) -> Tensor<B, 1> {
    let note_duration = PREAMBLE_NOTE_DURATION;
    let mut sequence = Vec::new();
    for _ in 0..cycles {
        sequence.extend(get_shifted_sweep_down(0));
    }
    generate_from_sequence::<B>(device, &sequence, note_duration)
}

fn generate_from_sequence<B: Backend>(device: &B::Device, sequence: &[usize], note_duration: f64) -> Tensor<B, 1> {
    // Generate each note
    let mut waveforms = Vec::new();
    for &idx in sequence {
        let waveform = generate_symbol::<B>(device, idx, 0.0, note_duration, FS);
        waveforms.push(waveform);
    }
    
    // Concatenate all waveforms
    Tensor::cat(waveforms, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    type TestBackend = Wgpu;
    
    #[test]
    fn test_morlet_wavelet() {
        let device = Default::default();
        let (real, imag) = morlet_wavelet::<TestBackend>(&device, 440.0, 2.0, 8000.0);
        
        // Check dimensions
        assert_eq!(real.dims()[0], 16000); // 2.0s * 8000Hz
        assert_eq!(imag.dims()[0], 16000);
        
        println!("Morlet wavelet generated successfully");
    }
    
    #[test]
    fn test_generate_symbol() {
        let device = Default::default();
        let waveform = generate_symbol::<TestBackend>(&device, 5, 0.0, 2.0, 8000.0);
        
        assert_eq!(waveform.dims()[0], 16000);
        println!("Symbol generated successfully");
    }
    
    #[test]
    fn test_bach_preamble() {
        let device = Default::default();
        let preamble = generate_bach_preamble::<TestBackend>(&device);
        
        // Should be 4 * 16 * 0.05s * 8000Hz
        let expected_len = 4 * 16 * (PREAMBLE_NOTE_DURATION * FS) as usize;
        assert_eq!(preamble.dims()[0], expected_len);
        
        println!("Bach preamble generated successfully");
    }
}
