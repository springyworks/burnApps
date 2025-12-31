use burn::tensor::{Tensor, backend::Backend, ElementConversion};

/// Compute cross-correlation using GPU-accelerated matrix multiplication
/// 
/// signal: [Length]
/// reference: [RefLength]
/// Returns: [Length - RefLength + 1] correlation values
pub fn cross_correlation_gpu<B: Backend>(
    device: &B::Device,
    signal: &Tensor<B, 1>,
    reference: &Tensor<B, 1>,
) -> Tensor<B, 1> {
    let sig_len = signal.dims()[0];
    let ref_len = reference.dims()[0];
    
    if sig_len < ref_len {
        return Tensor::zeros([1], device);
    }
    
    // Use batched matrix multiplication for correlation
    // This keeps everything on GPU and is efficient
    
    let output_len = sig_len - ref_len + 1;
    let max_delay = output_len; 
    
    // Create a batch of shifted signals.
    // [Batch=MaxDelay, Length=RefLen]
    let mut segments = Vec::new();
    for delay in 0..max_delay {
        let segment = signal.clone().slice([delay..delay + ref_len]);
        segments.push(segment);
    }
    
    // Stack into [MaxDelay, RefLen]
    let batch = Tensor::stack(segments, 0);
    
    // Reference: [RefLen] -> [RefLen, 1]
    let ref_col = reference.clone().reshape([ref_len, 1]);
    
    // Matmul: [MaxDelay, RefLen] @ [RefLen, 1] -> [MaxDelay, 1]
    let correlations = batch.matmul(ref_col).reshape([max_delay]);
    
    correlations
}

/// Soft combine LLRs from multiple repetitions (Maximum Ratio Combining)
/// 
/// llrs: [NumReps, NumBits]
/// weights: [NumReps] (SNR-based weights)
/// Returns: [NumBits] combined LLRs
pub fn soft_combine_gpu<B: Backend>(
    llrs: &Tensor<B, 2>,
    weights: &Tensor<B, 1>,
) -> Tensor<B, 1> {
    // Maximum Ratio Combining: weighted sum by SNR
    // weights: [N] -> [N, 1]
    let w = weights.clone().reshape([weights.dims()[0], 1]);
    
    // llrs * w broadcasts along dim 1
    // sum along dim 0
    (llrs.clone() * w).sum_dim(0).reshape([llrs.dims()[1]])
}

/// Coherent combining with phase alignment
/// 
/// Combines complex symbols from multiple repetitions with phase tracking
/// symbols_real: [NumReps, NumSymbols] real part
/// symbols_imag: [NumReps, NumSymbols] imaginary part  
/// Returns: ([NumSymbols] real, [NumSymbols] imag) combined
pub fn coherent_combine_symbols<B: Backend>(
    symbols_real: &Tensor<B, 2>,
    symbols_imag: &Tensor<B, 2>,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let num_reps = symbols_real.dims()[0];
    
    // Phase alignment: use first repetition as reference
    let ref_real = symbols_real.clone().slice([0..1, 0..symbols_real.dims()[1]]).reshape([symbols_real.dims()[1]]);
    let ref_imag = symbols_imag.clone().slice([0..1, 0..symbols_imag.dims()[1]]).reshape([symbols_imag.dims()[1]]);
    
    let mut aligned_real = vec![ref_real.clone()];
    let mut aligned_imag = vec![ref_imag.clone()];
    
    // Align remaining repetitions to reference
    for i in 1..num_reps {
        let curr_real = symbols_real.clone().slice([i..i+1, 0..symbols_real.dims()[1]]).reshape([symbols_real.dims()[1]]);
        let curr_imag = symbols_imag.clone().slice([i..i+1, 0..symbols_imag.dims()[1]]).reshape([symbols_imag.dims()[1]]);
        
        // Compute cross-correlation phase offset
        // conj(ref) * curr = (ref_r - j*ref_i) * (curr_r + j*curr_i)
        //                  = (ref_r*curr_r + ref_i*curr_i) + j*(ref_r*curr_i - ref_i*curr_r)
        let corr_real = ref_real.clone() * curr_real.clone() + ref_imag.clone() * curr_imag.clone();
        let corr_imag = ref_real.clone() * curr_imag.clone() - ref_imag.clone() * curr_real.clone();
        
        // Average phase (stay on GPU)
        let phase_real_avg = corr_real.mean();
        let phase_imag_avg = corr_imag.mean();
        
        // Normalize phase correction (GPU tensors)
        let phase_mag = (phase_real_avg.clone().powf_scalar(2.0) + phase_imag_avg.clone().powf_scalar(2.0)).sqrt().clamp_min(1e-10);
        let cos_theta = phase_real_avg / phase_mag.clone();
        let sin_theta = phase_imag_avg / phase_mag;
        
        // Rotate current symbol: curr * conj(phase) = curr * (cos - j*sin)
        let rotated_real = curr_real.clone() * cos_theta.clone() + curr_imag.clone() * sin_theta.clone();
        let rotated_imag = curr_imag * cos_theta - curr_real * sin_theta;
        
        aligned_real.push(rotated_real);
        aligned_imag.push(rotated_imag);
    }
    
    // Equal gain combining (since phases are aligned)
    let combined_real = Tensor::stack(aligned_real, 0).mean_dim(0);
    let combined_imag = Tensor::stack(aligned_imag, 0).mean_dim(0);
    
    (combined_real, combined_imag)
}

/// Estimate SNR from correlation peaks - GPU-only version
/// 
/// **NO SYNC POINT**: Returns tensor SNR in dB
/// Caller decides when to convert to scalar
pub fn estimate_snr_from_correlation_gpu<B: Backend>(
    correlation: &Tensor<B, 1>,
    peak_idx: usize,
    noise_window: usize,
) -> Tensor<B, 1> {
    let corr_len = correlation.dims()[0];
    let window_start = peak_idx.saturating_sub(noise_window);
    let window_end = (peak_idx + noise_window).min(corr_len);
    
    // Signal power from peak (GPU tensor)
    let peak_val = correlation.clone().slice([peak_idx..peak_idx+1]);
    let signal_power = peak_val.powf_scalar(2.0);
    
    // Noise power from regions away from peak
    let device = correlation.device();
    let mut noise_power_sum = Tensor::zeros([1], &device);
    let mut noise_count = 0;
    
    // Left side
    if window_start > 0 {
        let left_noise = correlation.clone().slice([0..window_start]).powf_scalar(2.0);
        noise_power_sum = noise_power_sum + left_noise.sum();
        noise_count += window_start;
    }
    
    // Right side
    if window_end < corr_len {
        let right_noise = correlation.clone().slice([window_end..corr_len]).powf_scalar(2.0);
        noise_power_sum = noise_power_sum + right_noise.sum();
        noise_count += corr_len - window_end;
    }
    
    if noise_count == 0 {
        return Tensor::from_floats([10.0], &device); // Default
    }
    
    let noise_power = (noise_power_sum / noise_count as f32).clamp_min(1e-10);
    let snr_linear = signal_power / noise_power;
    
    // SNR in dB (stays on GPU)
    snr_linear.log() * 10.0 / 2.302585 // log10(x) = ln(x) / ln(10)
}

/// Estimate SNR from correlation peaks
/// ⚠️ **SYNC POINT**: Returns scalar f32, downloads from GPU
/// 
/// For GPU pipelines, use estimate_snr_from_correlation_gpu() instead
pub fn estimate_snr_from_correlation<B: Backend>(
    correlation: &Tensor<B, 1>,
    peak_idx: usize,
    noise_window: usize,
) -> f32 {
    let snr_tensor = estimate_snr_from_correlation_gpu(correlation, peak_idx, noise_window);
    snr_tensor.into_scalar().elem()
}
