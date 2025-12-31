/// FFT-based cross-correlation using the convolution theorem
/// 
/// This is O(N log N) instead of O(N*M) for sliding window correlation.
/// 
/// Convolution theorem: correlation(signal, preamble) = IFFT(FFT(signal) × conj(FFT(preamble)))

use burn::tensor::{Tensor, backend::Backend};

// Re-export FftBackend trait so users can import it
pub use fft_gpu::cube_fft::FftBackend;

/// Compute cross-correlation using FFT (GPU-accelerated)
/// 
/// signal: [N] samples
/// reference: [M] samples where M <= N
/// 
/// Returns: [N - M + 1] correlation values
/// 
/// **Performance**: O(N log N) instead of O(N*M)
/// **No CPU sync** until you call .to_data() on result
pub fn fft_cross_correlation<B: Backend + FftBackend>(
    device: &B::Device,
    signal: &Tensor<B, 1>,
    reference: &Tensor<B, 1>,
) -> Tensor<B, 1> {
    let sig_len = signal.dims()[0];
    let ref_len = reference.dims()[0];
    
    if sig_len < ref_len {
        return Tensor::zeros([1], device);
    }
    
    // Find next power of 2 >= sig_len for FFT
    let fft_size = sig_len.next_power_of_two();
    
    // 1. Zero-pad both signals to FFT size
    let signal_padded = if sig_len < fft_size {
        let zeros = Tensor::zeros([fft_size - sig_len], device);
        Tensor::cat(vec![signal.clone(), zeros], 0)
    } else {
        signal.clone()
    };
    
    let reference_padded = if ref_len < fft_size {
        let zeros = Tensor::zeros([fft_size - ref_len], device);
        Tensor::cat(vec![reference.clone(), zeros], 0)
    } else {
        reference.clone()
    };
    
    // 2. Reshape to [1, fft_size] for batch FFT
    let sig_batch = signal_padded.reshape([1, fft_size]);
    let ref_batch = reference_padded.reshape([1, fft_size]);
    
    // 3. Convert to complex (real, imag=0)
    let sig_real = sig_batch;
    let sig_imag: Tensor<B, 2> = Tensor::zeros([1, fft_size], device);
    
    let ref_real = ref_batch;
    let ref_imag: Tensor<B, 2> = Tensor::zeros([1, fft_size], device);
    
    // 4. Forward FFT on both signals
    let sig_real_prim = sig_real.into_primitive();
    let sig_imag_prim = sig_imag.into_primitive();
    
    let sig_real_t = match sig_real_prim {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    let sig_imag_t = match sig_imag_prim {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    
    let (sig_fft_real_t, sig_fft_imag_t) = B::fft_1d_batch_impl(sig_real_t, sig_imag_t, fft_size);
    
    let ref_real_prim = ref_real.into_primitive();
    let ref_imag_prim = ref_imag.into_primitive();
    
    let ref_real_t = match ref_real_prim {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    let ref_imag_t = match ref_imag_prim {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    
    let (ref_fft_real_t, ref_fft_imag_t) = B::fft_1d_batch_impl(ref_real_t, ref_imag_t, fft_size);
    
    // Wrap back to Tensors
    let sig_fft_real: Tensor<B, 2> = Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(sig_fft_real_t));
    let sig_fft_imag: Tensor<B, 2> = Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(sig_fft_imag_t));
    let ref_fft_real: Tensor<B, 2> = Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(ref_fft_real_t));
    let ref_fft_imag: Tensor<B, 2> = Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(ref_fft_imag_t));
    
    // 5. Complex multiplication: signal_fft × conj(ref_fft)
    // (a + bi) × (c - di) = (ac + bd) + (bc - ad)i
    let prod_real = sig_fft_real.clone().mul(ref_fft_real.clone())
        .add(sig_fft_imag.clone().mul(ref_fft_imag.clone()));
    
    let prod_imag = sig_fft_imag.mul(ref_fft_real)
        .sub(sig_fft_real.mul(ref_fft_imag));
    
    // 6. Inverse FFT (same as forward FFT, then scale and flip imaginary)
    let prod_real_prim = prod_real.into_primitive();
    let prod_imag_prim = prod_imag.into_primitive();
    
    let prod_real_t = match prod_real_prim {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    let prod_imag_t = match prod_imag_prim {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    
    // IFFT = FFT with negated imaginary part, then scale by 1/N
    let prod_imag_neg_t = match Tensor::<B, 2>::from_primitive(burn::tensor::TensorPrimitive::Float(prod_imag_t))
        .neg()
        .into_primitive() {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    
    let (ifft_real_t, _ifft_imag_t) = B::fft_1d_batch_impl(prod_real_t, prod_imag_neg_t, fft_size);
    
    let mut correlation: Tensor<B, 2> = Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(ifft_real_t));
    
    // Scale by 1/N
    correlation = correlation.div_scalar(fft_size as f32);
    
    // 7. Extract valid correlation values [0..sig_len - ref_len + 1]
    let output_len = sig_len - ref_len + 1;
    let correlation_1d = correlation.reshape([fft_size]);
    
    correlation_1d.slice([0..output_len])
}

/// Convenience wrapper that works like the old cross_correlation_gpu
/// but uses FFT internally (much faster!)
pub fn cross_correlation_fft<B: Backend + FftBackend>(
    device: &B::Device,
    signal: &Tensor<B, 1>,
    reference: &Tensor<B, 1>,
) -> Tensor<B, 1> {
    fft_cross_correlation(device, signal, reference)
}
