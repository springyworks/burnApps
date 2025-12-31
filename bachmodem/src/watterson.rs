/// Watterson HF channel model simulation
/// 
/// Simulates realistic HF ionospheric propagation with:
/// - Multipath fading (2-3 paths with different delays)
/// - Doppler spread from ionospheric motion
/// - Frequency-selective fading
/// 
/// Reference: ITU-R Rec. F.1487, "Testing of HF modems with bandwidths of up to about 12 kHz using ionospheric channel simulators"

use burn::tensor::{Tensor, Distribution, backend::Backend};
use std::f32::consts::PI;

/// Watterson channel configuration
pub struct WattersonChannel {
    /// Number of propagation paths (typically 2-3)
    pub num_paths: usize,
    
    /// Path delays in samples (0, ~8ms, ~16ms for moderate spread)
    pub path_delays: Vec<usize>,
    
    /// Path gains (linear, typically exponential decay)
    pub path_gains: Vec<f32>,
    
    /// Doppler spread in Hz (0.1-2.0 Hz typical for HF)
    pub doppler_spread: f32,
    
    /// Sampling rate
    pub sample_rate: f32,
}

impl WattersonChannel {
    /// Create moderate HF channel (ITU Poor channel)
    pub fn moderate() -> Self {
        Self {
            num_paths: 2,
            path_delays: vec![0, 64],  // 0ms and 8ms at 8kHz
            path_gains: vec![0.7, 0.3], // -3 dB and -10 dB
            doppler_spread: 1.0,        // 1 Hz spread
            sample_rate: 8000.0,
        }
    }
    
    /// Create severe HF channel (ITU Very Poor)
    pub fn severe() -> Self {
        Self {
            num_paths: 3,
            path_delays: vec![0, 64, 128],  // 0, 8ms, 16ms
            path_gains: vec![0.6, 0.3, 0.1], // -4, -10, -20 dB
            doppler_spread: 2.0,             // 2 Hz spread
            sample_rate: 8000.0,
        }
    }
    
    /// Apply Watterson channel to signal
    pub fn apply<B: Backend>(&self, device: &B::Device, signal: &Tensor<B, 1>) -> Tensor<B, 1> {
        let signal_len = signal.dims()[0];
        
        // Initialize output with zeros
        let mut output = Tensor::<B, 1>::zeros([signal_len], device);
        
        // For each propagation path
        for path_idx in 0..self.num_paths {
            let delay = self.path_delays[path_idx];
            let gain = self.path_gains[path_idx];
            
            // Generate Rayleigh fading for this path (Jakes model)
            let fading = self.generate_rayleigh_fading::<B>(device, signal_len);
            
            // Create delayed signal using pure tensor operations
            let delayed_signal = if delay == 0 {
                signal.clone()
            } else if delay >= signal_len {
                Tensor::zeros([signal_len], device)
            } else {
                // Zeros for the delay period
                let zeros = Tensor::zeros([delay], device);
                // Signal shifted by delay
                let signal_part = signal.clone().slice([0..(signal_len - delay)]);
                Tensor::cat(vec![zeros, signal_part], 0)
            };
            
            // Apply fading and gain
            // output += delayed_signal * fading * gain
            output = output + (delayed_signal * fading * gain);
        }
        
        output
    }
    
    /// Generate Rayleigh fading using Jakes model
    fn generate_rayleigh_fading<B: Backend>(&self, device: &B::Device, length: usize) -> Tensor<B, 1> {
        // Jakes model: sum of sinusoids with random phases
        let num_oscillators = 16; // More = better approximation
        let fd = self.doppler_spread;
        
        let mut fading = Tensor::<B, 1>::zeros([length], device);
        
        // Pre-calculate time vector on GPU
        let t = Tensor::<B, 1, burn::tensor::Int>::arange(0..length as i64, device)
            .float() / self.sample_rate;
        
        for n in 0..num_oscillators {
            // Random phase for each oscillator
            let phase = (n as f32 / num_oscillators as f32) * 2.0 * PI;
            
            // Doppler frequency for this oscillator
            let fn_doppler = fd * (2.0 * PI * n as f32 / num_oscillators as f32).cos();
            
            // Generate oscillation
            let angle = t.clone() * (2.0 * PI * fn_doppler) + phase;
            let oscillation = angle.cos() / (num_oscillators as f32).sqrt();
            
            fading = fading + oscillation;
        }
        
        // Convert to Rayleigh envelope (magnitude of complex Gaussian)
        // Simple approximation: abs(fading)
        fading.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    type TestBackend = Wgpu;
    
    #[test]
    fn test_watterson_moderate() {
        let device = Default::default();
        let channel = WattersonChannel::moderate();
        
        // Generate simple test signal
        let signal = Tensor::<TestBackend, 1>::ones([16000], &device);
        
        // Apply channel
        let output = channel.apply::<TestBackend>(&device, &signal);
        
        // Check output has same length
        assert_eq!(output.dims()[0], 16000);
        
        println!("Watterson moderate channel test passed");
    }
}
