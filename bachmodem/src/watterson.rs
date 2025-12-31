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
    /// Create gentle HF channel (Good propagation)
    pub fn gentle() -> Self {
        Self {
            num_paths: 2,
            path_delays: vec![0, 32],   // 0ms and 4ms at 8kHz
            path_gains: vec![0.8, 0.2], // -2 dB and -14 dB
            doppler_spread: 0.05,       // 0.05 Hz spread (Very gentle)
            sample_rate: 8000.0,
        }
    }
    
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
        
        // Use random phases to avoid "start at 0" issue
        // We need two independent processes for Rayleigh (I and Q)
        // But for amplitude modulation of real signal, one process with non-zero start is enough?
        // Actually, let's do it properly: Envelope = sqrt(I^2 + Q^2)
        
        let t = Tensor::<B, 1, burn::tensor::Int>::arange(0..length as i64, device)
            .float() / self.sample_rate;
            
        let mut i_comp = Tensor::<B, 1>::zeros([length], device);
        let mut q_comp = Tensor::<B, 1>::zeros([length], device);
        
        // We can't easily use rand inside the loop if we want to be pure, 
        // but for simulation it's fine.
        // We need to import rand.
        let mut rng = rand::thread_rng();
        use rand::Rng;
        
        for n in 0..num_oscillators {
            // Doppler frequency
            let fn_doppler = fd * (2.0 * PI * n as f32 / num_oscillators as f32).cos();
            let omega = 2.0 * PI * fn_doppler;
            
            // Random phases for I and Q
            let phase_i = rng.gen::<f32>() * 2.0 * PI;
            let phase_q = rng.gen::<f32>() * 2.0 * PI;
            
            // I component
            let angle_i = t.clone() * omega + phase_i;
            let osc_i = angle_i.cos();
            i_comp = i_comp + osc_i;
            
            // Q component
            let angle_q = t.clone() * omega + phase_q;
            let osc_q = angle_q.cos();
            q_comp = q_comp + osc_q;
        }
        
        // Normalize
        let norm = (num_oscillators as f32).sqrt();
        i_comp = i_comp / norm;
        q_comp = q_comp / norm;
        
        // Envelope = sqrt(I^2 + Q^2)
        (i_comp.powf_scalar(2.0) + q_comp.powf_scalar(2.0)).sqrt()
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
