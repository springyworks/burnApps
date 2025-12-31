/// RAKE Receiver for Multipath Exploitation
/// 
/// Instead of treating multipath as interference, the RAKE receiver
/// detects and combines multiple delayed signal copies to INCREASE SNR
/// 
/// Named after a garden rake - each "finger" collects energy from one path

use burn::tensor::{Tensor, backend::Backend, ElementConversion};
use crate::gpu_ops::cross_correlation_gpu;

/// RAKE finger - tracks one multipath component
#[derive(Clone, Debug)]
pub struct RakeFinger {
    /// Path delay in samples
    pub delay: usize,
    
    /// Path complex amplitude (magnitude)
    pub amplitude: f32,
    
    /// Path phase offset (radians)
    pub phase: f32,
    
    /// Finger weight (for combining)
    pub weight: f32,
}

/// RAKE receiver configuration
pub struct RakeReceiver {
    /// Number of fingers (paths to track)
    pub num_fingers: usize,
    
    /// Maximum path delay to search (samples)
    pub max_delay: usize,
    
    /// Active fingers
    pub fingers: Vec<RakeFinger>,
}

impl RakeReceiver {
    /// Create RAKE receiver
    pub fn new(num_fingers: usize, max_delay: usize) -> Self {
        Self {
            num_fingers,
            max_delay,
            fingers: Vec::new(),
        }
    }
    
    /// Detect multipath components using correlation
    /// ⚠️ Contains SYNC POINTS in peak-finding loop
    /// 
    /// TODO: Replace with GPU-native topk operation when available
    pub fn detect_paths<B: Backend>(
        &mut self,
        device: &<B as Backend>::Device,
        signal: &Tensor<B, 1>,
        reference: &Tensor<B, 1>,
    ) {
        let sig_len = signal.dims()[0];
        let ref_len = reference.dims()[0];
        
        if sig_len < ref_len {
            println!("  [RAKE] Signal too short for path detection");
            return;
        }
        
        // Compute correlation at different delays using GPU
        // We limit the search to max_delay or signal length
        let search_len = self.max_delay.min(sig_len - ref_len);
        
        // Slice signal to search area + ref_len
        let search_signal = signal.clone().slice([0..search_len + ref_len]);
        
        // Compute all correlations in one go on GPU
        let correlations_tensor = cross_correlation_gpu(device, &search_signal, reference);
        
        // Find top peaks on GPU using iterative argmax
        self.fingers.clear();
        let mut remaining_corr = correlations_tensor.clone();
        
        for _ in 0..self.num_fingers {
            // ⚠️ SYNC POINT: Extract peak value
            // TODO: Use GPU topk/nlargest when Burn adds it
            let max_val: f32 = remaining_corr.clone().max().into_scalar().elem();
            
            if max_val < 0.1 {
                break; // No more significant peaks
            }
            
            // ⚠️ SYNC POINT: Extract peak index
            let argmax_val: i64 = remaining_corr.clone().argmax(0).into_scalar().elem();
            let delay = argmax_val as usize;
            
            let finger = RakeFinger {
                delay,
                amplitude: max_val,
                phase: 0.0, // Simplified: assume zero phase
                weight: max_val.abs(), // MRC weighting
            };
            
            self.fingers.push(finger);
            
            // Zero out region around this peak to find next peak
            let suppress_start = delay.saturating_sub(5);
            let suppress_end = (delay + 5).min(remaining_corr.dims()[0]);
            
            if suppress_end > suppress_start {
                let zeros = Tensor::zeros([suppress_end - suppress_start], device);
                remaining_corr = remaining_corr.slice_assign([suppress_start..suppress_end], zeros);
            }
        }
        
        println!("  [RAKE] Detected {} paths:", self.fingers.len());
        for (i, finger) in self.fingers.iter().enumerate() {
            println!("    Finger {}: delay={}samples ({:.2}ms), amp={:.3}",
                i, finger.delay, finger.delay as f32 / 8.0, finger.amplitude);
        }
    }
    
    /// Combine multipath components using Maximum Ratio Combining (MRC)
    pub fn combine_paths<B: Backend>(
        &self,
        device: &<B as Backend>::Device,
        signal: &Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        if self.fingers.is_empty() {
            println!("  [RAKE] No fingers, returning original signal");
            return signal.clone();
        }
        
        let sig_len = signal.dims()[0];
        
        // Find minimum output length (limited by longest delay)
        let max_delay = self.fingers.iter().map(|f| f.delay).max().unwrap_or(0);
        let output_len = sig_len.saturating_sub(max_delay);
        
        if output_len < 1000 {
            println!("  [RAKE] Output too short after delay compensation");
            return signal.clone();
        }
        
        // Initialize combined output
        let mut combined = Tensor::<B, 1>::zeros([output_len], device);
        
        // Sum of weights for normalization
        let total_weight: f32 = self.fingers.iter().map(|f| f.weight).sum();
        
        // Combine each finger's contribution
        for finger in &self.fingers {
            // Extract delayed signal
            let start = finger.delay;
            let end = start + output_len;
            
            if end <= sig_len {
                let delayed = signal.clone().slice([start..end]);
                
                // Weight by finger strength (MRC)
                let weighted = delayed * (finger.weight / total_weight);
                
                // Add to combined output
                combined = combined + weighted;
            }
        }
        
        println!("  [RAKE] Combined {} paths with MRC", self.fingers.len());
        
        combined
    }
    
    /// Simplified RAKE processing (detect + combine)
    pub fn process<B: Backend>(
        &mut self,
        device: &<B as Backend>::Device,
        signal: &Tensor<B, 1>,
        reference: &Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        // Detect multipath
        self.detect_paths::<B>(device, signal, reference);
        
        // Combine paths
        self.combine_paths::<B>(device, signal)
    }
}

/// Estimate multipath gain from RAKE combining
/// Returns improvement in dB
pub fn estimate_rake_gain(num_paths: usize, path_powers: &[f32]) -> f32 {
    if num_paths <= 1 || path_powers.is_empty() {
        return 0.0;
    }
    
    // Total power = sum of all path powers
    let total_power: f32 = path_powers.iter().take(num_paths).sum();
    
    // Strongest path power
    let strongest_path = path_powers.iter()
        .take(num_paths)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&1.0);
    
    // Gain = total_power / strongest_path (in linear)
    let gain_linear = total_power / strongest_path;
    
    // Convert to dB
    10.0 * gain_linear.log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rake_gain_estimation() {
        // Example: 3 paths with relative powers 1.0, 0.5, 0.3
        let path_powers = vec![1.0, 0.5, 0.3];
        
        let gain = estimate_rake_gain(3, &path_powers);
        
        // Expected: 10*log10(1.8/1.0) ≈ 2.55 dB
        println!("RAKE gain: {:.2} dB", gain);
        assert!(gain > 2.0 && gain < 3.0);
    }
}
