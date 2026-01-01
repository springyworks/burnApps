/// Time-Slotted Repetition Protocol for BachModem
/// 
/// Features:
/// - Multiple repetitions of the same message
/// - Fixed time slots with listening gaps
/// - Multi-copy combining for improved SNR
/// - Coherent and non-coherent integration
/// - Multipath mitigation via diversity

use burn::tensor::{Tensor, backend::Backend, ElementConversion};
use crate::modulation::{modulate_fhdpsk_with_flourishes, encode_bits};
use crate::wavelet::{FS, SYMBOL_DURATION};

/// Time slot configuration for repetition protocol
#[derive(Clone, Debug)]
pub struct TimeSlotConfig {
    /// Duration of one complete transmission (seconds)
    pub transmission_duration: f64,
    
    /// Listening gap between transmissions (seconds)
    pub listening_gap: f64,
    
    /// Total number of repetitions
    pub num_repetitions: usize,
    
    /// Time slot start times (seconds from beginning)
    pub slot_starts: Vec<f64>,
}

impl TimeSlotConfig {
    /// Create time slot configuration for given message length
    pub fn new(message_bytes: usize, num_repetitions: usize, listening_gap: f64) -> Self {
        // Calculate transmission duration
        // Each byte = 8 bits, pad to multiple of 16, add reference block
        let total_bits = ((message_bytes * 8 + 15) / 16) * 16 + 16;
        let num_symbols = total_bits;
        let flourish_interval = 64;
        let num_flourishes = num_symbols / flourish_interval;
        
        let data_duration = num_symbols as f64 * SYMBOL_DURATION;
        let flourish_duration = num_flourishes as f64 * 1.6; // 1.6 seconds each (2 cycles * 16 notes * 0.05)
        let preamble_duration = 3.2; // 3.2 seconds (4 cycles * 16 notes * 0.05)
        let postamble_duration = 1.6; // 1.6 seconds (2 cycles * 16 notes * 0.05)
        
        let transmission_duration = preamble_duration + data_duration + flourish_duration + postamble_duration;
        
        // Calculate slot start times
        let mut slot_starts = Vec::new();
        let mut current_time = 0.0;
        
        for _ in 0..num_repetitions {
            slot_starts.push(current_time);
            current_time += transmission_duration + listening_gap;
        }
        
        Self {
            transmission_duration,
            listening_gap,
            num_repetitions,
            slot_starts,
        }
    }
    
    /// Total duration including all repetitions and gaps
    pub fn total_duration(&self) -> f64 {
        if self.num_repetitions == 0 {
            return 0.0;
        }
        self.slot_starts[self.num_repetitions - 1] + self.transmission_duration
    }
}

/// Generate time-slotted repetition transmission
pub fn generate_repetition_transmission<B: Backend>(
    device: &B::Device,
    message: &[u8],
    config: &TimeSlotConfig,
) -> Tensor<B, 1> {
    // Generate one clean transmission
    let single_transmission = modulate_fhdpsk_with_flourishes::<B>(
        device,
        message,
        true,  // Add preamble
        32,    // Flourish interval (more frequent inter-ambles)
    );
    
    let transmission_len = single_transmission.dims()[0];
    let gap_len = (config.listening_gap * FS) as usize;
    
    // Create empty buffer for all repetitions
    let total_samples = (config.total_duration() * FS) as usize;
    let mut output = Tensor::<B, 1>::zeros([total_samples], device);
    
    for (rep_idx, &slot_start) in config.slot_starts.iter().enumerate() {
        let start_sample = (slot_start * FS) as usize;
        
        println!("  Repetition {}/{}: starts at {:.1}s (sample {})", 
            rep_idx + 1, config.num_repetitions, slot_start, start_sample);
        
        // Copy transmission into this slot
        let end_sample = (start_sample + transmission_len).min(total_samples);
        let len = end_sample.saturating_sub(start_sample);
        
        if len > 0 {
            // Use slice_assign to copy data on GPU
            let values = single_transmission.clone().slice([0..len]);
            output = output.slice_assign([start_sample..start_sample+len], values);
        }
    }
    
    output
}

/// Multi-copy combining strategies
#[derive(Clone, Copy, Debug)]
pub enum CombiningStrategy {
    /// Select best single copy (highest SNR estimate)
    SelectBest,
    
    /// Non-coherent combining (add magnitudes)
    NonCoherent,
    
    /// Coherent combining (add with phase alignment)
    Coherent,
    
    /// Maximum Ratio Combining (weight by SNR estimate)
    MaxRatio,
}

/// Decoded copy with quality metrics
#[derive(Clone, Debug)]
pub struct DecodedCopy {
    /// Repetition index
    pub repetition: usize,
    
    /// Decoded bytes
    pub data: Vec<u8>,
    
    /// Estimated SNR (from correlation peak)
    pub snr_estimate: f32,
    
    /// Preamble correlation strength
    pub correlation: f32,
    
    /// Number of symbols extracted
    pub num_symbols: usize,
}

/// Combine multiple decoded copies using voting
pub fn combine_decoded_copies(copies: &[DecodedCopy]) -> Vec<u8> {
    if copies.is_empty() {
        return Vec::new();
    }
    
    // Find maximum length
    let max_len = copies.iter().map(|c| c.data.len()).max().unwrap_or(0);
    
    if max_len == 0 {
        return Vec::new();
    }
    
    let mut combined = Vec::new();
    
    // For each byte position
    for byte_idx in 0..max_len {
        // Collect all valid bytes at this position with their SNR weights
        let mut byte_votes: std::collections::HashMap<u8, f32> = std::collections::HashMap::new();
        
        for copy in copies {
            if byte_idx < copy.data.len() {
                let byte_val = copy.data[byte_idx];
                let weight = copy.snr_estimate.max(0.1); // Minimum weight
                *byte_votes.entry(byte_val).or_insert(0.0) += weight;
            }
        }
        
        // Select byte with highest weighted vote
        let best_byte = byte_votes.iter()
            .max_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap())
            .map(|(b, _)| *b)
            .unwrap_or(0);
        
        combined.push(best_byte);
    }
    
    combined
}

/// Multipath mitigation using frequency diversity
/// 
/// The idea: Different frequencies fade independently in multipath.
/// By decoding multiple frequency bins and combining, we get diversity gain.
pub fn estimate_frequency_diversity<B: Backend>(
    device: &B::Device,
    signal: &Tensor<B, 1>,
) -> Vec<f32> {
    // Estimate power in each of the 16 Bach frequency bins
    // This gives us an idea of which frequencies are faded
    
    let signal_len = signal.dims()[0];
    let window_len = 16000; // 2 second window
    
    let mut frequency_powers = vec![0.0f32; 16];
    
    // Simple power estimation (in practice would use FFT)
    // For now just return uniform - full implementation would analyze spectrum
    for i in 0..16 {
        frequency_powers[i] = 1.0;
    }
    
    frequency_powers
}

/// Time diversity via repeat combining
/// 
/// Multipath changes over time (seconds to minutes).
/// By repeating transmissions with gaps, we get time diversity.
pub fn estimate_time_diversity(repetitions: &[DecodedCopy]) -> f32 {
    if repetitions.len() < 2 {
        return 1.0;
    }
    
    // Measure variation in SNR across repetitions
    let snrs: Vec<f32> = repetitions.iter().map(|r| r.snr_estimate).collect();
    let mean_snr = snrs.iter().sum::<f32>() / snrs.len() as f32;
    
    let variance = snrs.iter()
        .map(|s| (s - mean_snr).powi(2))
        .sum::<f32>() / snrs.len() as f32;
    
    let std_dev = variance.sqrt();
    
    // High variance = good time diversity (channel changing)
    // Low variance = poor time diversity (stable channel)
    1.0 + (std_dev / mean_snr.max(0.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_slot_config() {
        let config = TimeSlotConfig::new(13, 5, 10.0);
        
        assert_eq!(config.num_repetitions, 5);
        assert_eq!(config.listening_gap, 10.0);
        assert!(config.transmission_duration > 0.0);
        assert_eq!(config.slot_starts.len(), 5);
        
        println!("Time slot config: {:#?}", config);
    }
    
    #[test]
    fn test_combine_decoded_copies() {
        let copies = vec![
            DecodedCopy {
                repetition: 0,
                data: b"Hello".to_vec(),
                snr_estimate: 10.0,
                correlation: 0.8,
                num_symbols: 50,
            },
            DecodedCopy {
                repetition: 1,
                data: b"Hallo".to_vec(), // One error
                snr_estimate: 5.0,
                correlation: 0.6,
                num_symbols: 50,
            },
            DecodedCopy {
                repetition: 2,
                data: b"Hello".to_vec(),
                snr_estimate: 8.0,
                correlation: 0.7,
                num_symbols: 50,
            },
        ];
        
        let combined = combine_decoded_copies(&copies);
        
        // Should vote for "Hello" (2 high-SNR votes vs 1 low-SNR vote)
        assert_eq!(combined, b"Hello");
        
        println!("Combined result: {:?}", String::from_utf8_lossy(&combined));
    }
}
