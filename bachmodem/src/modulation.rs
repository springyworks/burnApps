use burn::tensor::{Tensor, backend::Backend, ElementConversion};
use crate::wavelet::{generate_symbol, generate_bach_preamble, generate_bach_flourish, get_melody_indices, morlet_wavelet, FS, SYMBOL_DURATION, BACH_FREQUENCIES};
use crate::gpu_ops::cross_correlation_gpu;
use crate::fft_correlation::{fft_cross_correlation, FftBackend};
use crate::gpu_math::atan2_fast_gpu;
use std::f64::consts::PI;

/// Encodes bytes into a sequence of bits
pub fn encode_bits(data_bytes: &[u8]) -> Vec<u8> {
    data_bytes
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect()
}

/// Packs bits back into bytes
pub fn pack_bits(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0u8, |acc, (i, &bit)| {
                acc | (bit << (7 - i))
            })
        })
        .collect()
}

/// Modulates data using Frequency-Hopping Differential Phase Shift Keying (FH-DPSK)
pub fn modulate_fhdpsk<B: Backend>(
    device: &B::Device,
    data_bytes: &[u8],
    add_preamble: bool,
) -> Tensor<B, 1> {
    modulate_fhdpsk_with_flourishes::<B>(device, data_bytes, add_preamble, 0)
}

/// Modulates with periodic musical flourishes (Bach Sweeps) inserted throughout
pub fn modulate_fhdpsk_with_flourishes<B: Backend>(
    device: &B::Device,
    data_bytes: &[u8],
    add_preamble: bool,
    flourish_interval: usize, // Insert flourish every N symbols (0 = disabled)
) -> Tensor<B, 1> {
    let bits = encode_bits(data_bytes);
    
    if bits.is_empty() {
        if add_preamble {
            return generate_bach_preamble::<B>(device);
        } else {
            return Tensor::from_floats([0.0f32], device);
        }
    }
    
    // Pad bits to multiple of 16 for block processing
    let mut padded_bits = bits.clone();
    let pad_len = (16 - (bits.len() % 16)) % 16;
    padded_bits.extend(vec![0; pad_len]);
    
    // Prepend reference block (16 zeros) to establish phase reference
    let mut bits_with_ref = vec![0u8; 16];
    bits_with_ref.extend(padded_bits);
    
    // Reshape for Inter-Hop Differential Encoding (Lag 16)
    let num_blocks = bits_with_ref.len() / 16;
    let mut phases = Vec::new();
    
    // Cumulative sum along time axis for differential encoding
    for block_idx in 0..num_blocks {
        let block_start = block_idx * 16;
        for freq_idx in 0..16 {
            let bit_idx = block_start + freq_idx;
            let bit = bits_with_ref[bit_idx];
            
            // Phase shift: bit * π
            let phase_shift = if bit == 1 { PI } else { 0.0 };
            
            // Cumulative phase for this frequency
            let prev_phase = if block_idx == 0 {
                0.0
            } else {
                phases[(block_idx - 1) * 16 + freq_idx]
            };
            
            phases.push(prev_phase + phase_shift);
        }
    }
    
    // Generate melody sequence
    let num_symbols = phases.len();
    let melody_indices = get_melody_indices(num_symbols);
    
    // Generate waveforms with optional musical flourishes
    let mut waveforms = Vec::new();
    
    for (i, &melody_idx) in melody_indices.iter().enumerate() {
        // Insert Bach Sweep flourish periodically (if enabled)
        if flourish_interval > 0 && i > 0 && i % flourish_interval == 0 {
            let flourish = generate_bach_flourish::<B>(device);
            waveforms.push(flourish);
        }
        
        let phase = phases[i];
        let waveform = generate_symbol::<B>(device, melody_idx, phase, SYMBOL_DURATION, FS);
        waveforms.push(waveform);
    }
    
    let data_waveform = Tensor::cat(waveforms, 0);
    
    if add_preamble {
        let preamble = generate_bach_preamble::<B>(device);
        Tensor::cat(vec![preamble, data_waveform], 0)
    } else {
        data_waveform
    }
}

/// GPU-only synchronization - returns tensors without sync
/// 
/// **NO SYNC POINT**: Returns (correlation_tensor, best_idx_tensor, best_val_tensor)
/// Caller decides when to sync. Use this in GPU pipelines.
/// 
/// **Now uses FFT-based correlation**: O(N log N) instead of O(N*M) - 100x+ faster!
pub fn synchronize_signal_gpu<B: Backend + FftBackend>(
    device: &B::Device,
    signal: &Tensor<B, 1>,
    preamble: &Tensor<B, 1>,
) -> (Tensor<B, 1>, Tensor<B, 1, burn::tensor::Int>, Tensor<B, 1>) {
    let correlations = fft_cross_correlation(device, signal, preamble);
    let (max_val, max_idx_tensor) = correlations.clone().max_dim_with_indices(0);
    
    (correlations, max_idx_tensor, max_val)
}

/// Synchronizes signal by finding the Bach Preamble via cross-correlation
/// ⚠️ **SYNC POINT**: Returns scalar position, downloads from GPU
/// 
/// For GPU-only pipelines, use synchronize_signal_gpu() instead
/// 
/// **Now uses FFT-based correlation**: O(N log N) instead of O(N*M) - 100x+ faster!
pub fn synchronize_signal<B: Backend + FftBackend>(
    device: &B::Device,
    signal: &Tensor<B, 1>,
) -> Option<usize> {
    println!("    [Sync] Starting synchronization...");
    let preamble = generate_bach_preamble::<B>(device);
    let preamble_len = preamble.dims()[0];
    let signal_len = signal.dims()[0];
    
    println!("    [Sync] Signal len: {}, Preamble len: {}", signal_len, preamble_len);
    
    if signal_len < preamble_len {
        println!("    [Sync] Signal too short!");
        return None;
    }
    
    // FFT correlation is fast enough - no need to decimate!
    // This avoids aliasing (max freq 1174 Hz > 1000 Hz Nyquist at decimation 4)
    let decim_factor = 1; 
    println!("    [Sync] No decimation for full accuracy...");
    
    let decim_signal_tensor = signal.clone();
    let decim_preamble_tensor = preamble.clone();
    
    println!("    [Sync] Coarse correlation: {} samples vs {} preamble", 
             decim_signal_tensor.dims()[0], decim_preamble_tensor.dims()[0]);
    
    // Coarse correlation on decimated signal
    let (correlations_coarse, _, _) = synchronize_signal_gpu(device, &decim_signal_tensor, &decim_preamble_tensor);
    
    println!("    [Sync] Finding coarse peak...");
    
    println!("    [Sync] Non-coherent integration (GPU-only for speed)...");
    // Square for non-coherent integration - STAY ON GPU to avoid CPU bottleneck
    let correlations_squared: Tensor<B, 1> = correlations_coarse.clone().powf_scalar(2.0);
    
    // Find max on GPU (avoids slow CPU download + sorting)
    let (max_val_tensor, max_idx_tensor) = correlations_squared.clone().max_dim_with_indices(0);
    let peak_val: f32 = max_val_tensor.into_scalar().elem::<f32>();
    let best_position_decim: usize = max_idx_tensor.into_scalar().elem::<i32>() as usize;
    let best_position = best_position_decim * decim_factor;
    
    // Fast noise floor estimate using mean (much faster than median sort on CPU)
    let mean_val: f32 = correlations_squared.clone().mean().into_scalar().elem::<f32>();
    let peak_to_noise_ratio = peak_val / (mean_val + 1e-10);
    
    // Extract metrics for threshold check  
    let (max_corr_tensor, _) = correlations_coarse.max_dim_with_indices(0);
    let max_corr_val: f32 = max_corr_tensor.into_scalar().elem::<f32>();
    let preamble_energy: f32 = decim_preamble_tensor.clone().powf_scalar(2.0).sum().into_scalar().elem::<f32>();
    let normalized_correlation = max_corr_val / preamble_energy.sqrt();
    
    println!("    [Sync] Corr: {:.4}, Peak: {:.6}, P/N: {:.2}, Pos: {}", 
             normalized_correlation, peak_val, peak_to_noise_ratio, best_position);

    // WSPR-style adaptive threshold for -30 dB (relaxed for 4x decimation)
    const CORRELATION_THRESHOLD: f32 = 0.025;  // Very aggressive for -30 dB
    const PEAK_TO_NOISE_THRESHOLD: f32 = 1.3; // Relaxed (weak signal)
    
    if normalized_correlation < CORRELATION_THRESHOLD {
        println!("    [Sync] Failed: correlation {:.4} < {:.4}", normalized_correlation, CORRELATION_THRESHOLD);
        return None;
    }
    
    if peak_to_noise_ratio < PEAK_TO_NOISE_THRESHOLD {
        println!("    [Sync] Failed: peak/noise {:.2} < {:.2}", peak_to_noise_ratio, PEAK_TO_NOISE_THRESHOLD);
        return None;
    }
    
    Some(best_position)
}

/// Demodulates FH-DPSK signal with proper synchronization and matched filtering
/// Set flourish_interval to the same value used during encoding (0 = no flourishes)
pub fn demodulate_fhdpsk_ex<B: Backend + FftBackend>(
    device: &B::Device,
    signal: &Tensor<B, 1>,
    use_sync: bool,
    flourish_interval: usize,
) -> Vec<u8> {
    let symbol_len = (SYMBOL_DURATION * FS) as usize;
    let flourish_len = generate_bach_flourish::<B>(device).dims()[0];
    
    let mut signal_data = signal.clone();
    
    if use_sync {
        // Find preamble via correlation
        match synchronize_signal::<B>(device, signal) {
            Some(sync_pos) => {
                println!("  [Decoder] Found preamble at position {}", sync_pos);
                
                let preamble_len = generate_bach_preamble::<B>(device).dims()[0];
                let start_pos = sync_pos + preamble_len;
                let signal_len = signal.dims()[0];
                
                if signal_len <= start_pos {
                    println!("  [Decoder] No data after preamble");
                    return Vec::new();
                }
                
                signal_data = signal.clone().slice([start_pos..signal_len]);
            }
            None => {
                println!("  [Decoder] Failed to find preamble!");
                return Vec::new();
            }
        }
    }
    
    let signal_len = signal_data.dims()[0];
    
    // Extract symbols, skipping flourishes at expected positions
    let mut symbol_chunks = Vec::new();
    let mut pos = 0;
    let mut symbol_idx = 0;
    
    while pos + symbol_len <= signal_len {
        // Check if we should skip a flourish here
        if flourish_interval > 0 && symbol_idx > 0 && symbol_idx % flourish_interval == 0 {
            // Skip the flourish
            pos += flourish_len;
            if pos + symbol_len > signal_len {
                break;
            }
            println!("  [Decoder] Skipping flourish at symbol position {}", symbol_idx);
        }
        
        let chunk = signal_data.clone().slice([pos..pos + symbol_len]);
        symbol_chunks.push(chunk);
        pos += symbol_len;
        symbol_idx += 1;
    }
    
    let num_symbols = symbol_chunks.len();
    
    if num_symbols == 0 {
        println!("  [Decoder] No symbols found");
        return Vec::new();
    }
    
    println!("  [Decoder] Extracted {} data symbols", num_symbols);
    
    // Matched filtering: correlate each symbol with expected wavelet
    let melody_indices = get_melody_indices(num_symbols);
    
    println!("  [Decoder] Performing matched filtering...");
    
    // Collect correlations on GPU
    let mut real_corrs = Vec::new();
    let mut imag_corrs = Vec::new();
    
    for (sym_idx, &melody_idx) in melody_indices.iter().enumerate() {
        // Generate reference wavelet (conjugate for correlation)
        let (real_ref, imag_ref) = morlet_wavelet::<B>(
            device,
            BACH_FREQUENCIES[melody_idx],
            SYMBOL_DURATION,
            FS,
        );
        
        // Get this symbol's chunk
        let chunk_1d = &symbol_chunks[sym_idx];
        
        // Correlate: sum(chunk * conj(ref)) - STAYS ON GPU
        let real_corr = (chunk_1d.clone().mul(real_ref)).sum();
        let imag_corr = (chunk_1d.clone().mul(imag_ref.neg())).sum(); // Conjugate
        
        real_corrs.push(real_corr);
        imag_corrs.push(imag_corr);
    }
    
    // Stack into tensors and compute atan2 on GPU (NO SYNC!)
    let real_tensor = Tensor::stack(real_corrs, 0);
    let imag_tensor = Tensor::stack(imag_corrs, 0);
    
    let angles_tensor = atan2_fast_gpu(imag_tensor, real_tensor);
    
    // Single sync at the end to get all angles
    let angles_data = angles_tensor.into_data();
    let mut correlations: Vec<f64> = angles_data.to_vec::<f32>().unwrap()
        .iter().map(|&x| x as f64).collect();
    
    // Differential decoding with Lag 16
    println!("  [Decoder] Differential decoding (Lag-16)...");
    
    let trunc_len = (correlations.len() / 16) * 16;
    correlations.truncate(trunc_len);
    
    if trunc_len < 32 {
        println!("  [Decoder] Insufficient symbols for decoding (need at least 32)");
        return Vec::new();
    }
    
    let num_blocks = trunc_len / 16;
    println!("  [Decoder] Processing {} blocks", num_blocks);
    
    // Calculate phase differences
    let mut detected_bits = Vec::new();
    
    for block_idx in 1..num_blocks {
        for freq_idx in 0..16 {
            let curr_idx = block_idx * 16 + freq_idx;
            let prev_idx = (block_idx - 1) * 16 + freq_idx;
            
            let mut diff = correlations[curr_idx] - correlations[prev_idx];
            
            // Wrap to [-π, π]
            while diff > PI {
                diff -= 2.0 * PI;
            }
            while diff < -PI {
                diff += 2.0 * PI;
            }
            
            // Decision: |diff| > π/2 => bit = 1
            let bit = if diff.abs() > PI / 2.0 { 1 } else { 0 };
            detected_bits.push(bit);
        }
    }
    
    // Note: The first differential block (block_1 - block_0) represents the FIRST 16 data bits
    // We don't need to remove anything - the reference block (block 0) is never output
    
    println!("  [Decoder] Decoded {} bits", detected_bits.len());

    
    let decoded_bytes = pack_bits(&detected_bits);
    println!("  [Decoder] Packed into {} bytes", decoded_bytes.len());
    
    decoded_bytes
}

/// Demodulates FH-DPSK signal returning Soft LLRs on GPU
/// 
/// Returns: Tensor of LLRs [NumBits]
/// Positive LLR -> Bit 0
/// Negative LLR -> Bit 1
pub fn demodulate_fhdpsk_soft<B: Backend + FftBackend>(
    device: &B::Device,
    signal: &Tensor<B, 1>,
    use_sync: bool,
    flourish_interval: usize,
) -> Tensor<B, 1> {
    let symbol_len = (SYMBOL_DURATION * FS) as usize;
    let flourish_len = generate_bach_flourish::<B>(device).dims()[0];
    
    let mut signal_data = signal.clone();
    
    if use_sync {
        match synchronize_signal::<B>(device, signal) {
            Some(sync_pos) => {
                let preamble_len = generate_bach_preamble::<B>(device).dims()[0];
                let start_pos = sync_pos + preamble_len;
                let signal_len = signal.dims()[0];
                if signal_len > start_pos {
                    signal_data = signal.clone().slice([start_pos..signal_len]);
                } else {
                    return Tensor::zeros([1], device); // Return dummy small tensor on failure
                }
            }
            None => return Tensor::zeros([1], device),
        }
    }
    
    let signal_len = signal_data.dims()[0];
    
    // 1. Extract Symbols into a Batch Tensor
    // We have to handle flourishes, so we can't just reshape.
    // We'll collect valid symbol segments.
    
    let mut segments = Vec::new();
    let mut pos = 0;
    let mut symbol_idx = 0;
    
    while pos + symbol_len <= signal_len {
        if flourish_interval > 0 && symbol_idx > 0 && symbol_idx % flourish_interval == 0 {
            pos += flourish_len;
            if pos + symbol_len > signal_len { break; }
        }
        
        let segment = signal_data.clone().slice([pos..pos + symbol_len]);
        segments.push(segment);
        pos += symbol_len;
        symbol_idx += 1;
    }
    
    let num_symbols = segments.len();
    if num_symbols == 0 { return Tensor::zeros([1], device); }
    
    // Stack: [NumSymbols, SymbolLen]
    let symbols_batch: Tensor<B, 2> = Tensor::stack(segments, 0);
    
    // 2. Matched Filtering on GPU
    // We need the reference wavelet for each symbol position.
    // The melody sequence is deterministic.
    let melody_indices = get_melody_indices(num_symbols);
    
    // Pre-generate all 16 unique wavelets: [16, SymbolLen]
    let mut unique_wavelets_real = Vec::new();
    let mut unique_wavelets_imag = Vec::new();
    
    for i in 0..16 {
        let (r, im) = morlet_wavelet::<B>(device, BACH_FREQUENCIES[i], SYMBOL_DURATION, FS);
        unique_wavelets_real.push(r);
        unique_wavelets_imag.push(im.neg()); // Conjugate for correlation
    }
    let bank_real: Tensor<B, 2> = Tensor::stack(unique_wavelets_real, 0);
    let bank_imag: Tensor<B, 2> = Tensor::stack(unique_wavelets_imag, 0);
    
    // Create indices tensor for gather
    // gather expects indices of same dim as output? No, gather is tricky in Burn.
    // Instead, let's just construct the reference batch by stacking clones (inefficient but easy)
    // or better: use matrix multiplication if we can map symbols to frequencies.
    
    // Let's construct the batch of references.
    // Since we are on CPU for the loop setup, we can just pick the right tensors from the bank.
    // Actually, we can't easily index into the bank tensor from CPU indices without pulling data.
    // But we have the bank tensors on GPU.
    // We can just rebuild the list of references.
    
    // Optimization: Group symbols by frequency? No, order matters.
    // Let's just stack the correct wavelets.
    let mut refs_real_list = Vec::with_capacity(num_symbols);
    let mut refs_imag_list = Vec::with_capacity(num_symbols);
    
    // We can reuse the bank slices to avoid re-generating, but we need to clone them to stack.
    // Burn tensors are cheap to clone (Arc).
    // But we need to slice the bank.
    // bank_real.slice([idx..idx+1])
    
    for &idx in &melody_indices {
        refs_real_list.push(bank_real.clone().slice([idx..idx+1]).reshape([symbol_len]));
        refs_imag_list.push(bank_imag.clone().slice([idx..idx+1]).reshape([symbol_len]));
    }
    
    let refs_real: Tensor<B, 2> = Tensor::stack(refs_real_list, 0); // [NumSymbols, SymbolLen]
    let refs_imag: Tensor<B, 2> = Tensor::stack(refs_imag_list, 0);
    
    // Dot product along dim 1
    // symbols_batch * refs
    let corr_real = (symbols_batch.clone() * refs_real).sum_dim(1).reshape([num_symbols]);
    let corr_imag = (symbols_batch * refs_imag).sum_dim(1).reshape([num_symbols]);
    
    // 3. Phase Extraction & Differential Decoding (Lag 16)
    // We avoid explicit atan2 by using trigonometric identities.
    // LLR = cos(angle_curr - angle_prev) * amplitude_curr
    // cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
    // cos(angle) = real / amp, sin(angle) = imag / amp
    // LLR = (real_curr*real_prev + imag_curr*imag_prev) / amp_prev
    
    let trunc_len = (num_symbols / 16) * 16;
    if trunc_len < 32 { return Tensor::zeros([1], device); }
    
    let corr_real_trunc = corr_real.slice([0..trunc_len]);
    let corr_imag_trunc = corr_imag.slice([0..trunc_len]);
    
    // Current symbols: start at index 16
    let real_curr = corr_real_trunc.clone().slice([16..trunc_len]);
    let imag_curr = corr_imag_trunc.clone().slice([16..trunc_len]);
    
    // Previous symbols: start at index 0, end at len-16
    let real_prev = corr_real_trunc.slice([0..trunc_len - 16]);
    let imag_prev = corr_imag_trunc.slice([0..trunc_len - 16]);
    
    // Amplitude of previous symbols
    let amp_prev = (real_prev.clone().powf_scalar(2.0) + imag_prev.clone().powf_scalar(2.0)).sqrt();
    
    // Dot product of phasors
    let dot_prod = real_curr * real_prev + imag_curr * imag_prev;
    
    // LLR calculation
    // Add epsilon to avoid division by zero
    let llrs = dot_prod / (amp_prev + 1e-6);
    
    llrs
}

/// Convenience wrapper for backwards compatibility
pub fn demodulate_fhdpsk<B: Backend + FftBackend>(
    device: &B::Device,
    signal: &Tensor<B, 1>,
    use_sync: bool,
) -> Vec<u8> {
    demodulate_fhdpsk_ex::<B>(device, signal, use_sync, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    type TestBackend = Wgpu;
    
    #[test]
    fn test_encode_bits() {
        let data = b"Hi";
        let bits = encode_bits(data);
        
        assert_eq!(bits.len(), 16);
        assert_eq!(&bits[0..8], &[0, 1, 0, 0, 1, 0, 0, 0]);
        assert_eq!(&bits[8..16], &[0, 1, 1, 0, 1, 0, 0, 1]);
    }
    
    #[test]
    fn test_modulate_fhdpsk() {
        let device = Default::default();
        let data = b"Test";
        let signal = modulate_fhdpsk::<TestBackend>(&device, data, false);
        
        let expected_symbols = 64;
        let expected_len = expected_symbols * (SYMBOL_DURATION * FS) as usize;
        
        println!("Signal length: {}, expected: {}", signal.dims()[0], expected_len);
        assert_eq!(signal.dims()[0], expected_len);
    }
}
