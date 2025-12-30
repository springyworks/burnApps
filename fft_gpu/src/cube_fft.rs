use cubecl::{cube, prelude::*};
use burn::tensor::{backend::Backend, ops::FloatTensor};
use burn_cubecl::{CubeBackend, CubeRuntime, FloatElement, IntElement, BoolElement, kernel::into_contiguous};
use burn_ndarray::{NdArray, NdArrayTensor};
use rustfft::{FftPlanner, num_complex::Complex};
use rustfft::num_traits::Zero;
use rayon::prelude::*;

#[cube]
fn reverse_bits(n: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    let mut x = n;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cube(launch)]
pub fn bit_reverse_kernel<F: Float>(
    real: &mut Tensor<F>,
    imag: &mut Tensor<F>,
    n_fft: u32,
    bits: u32,
) {
    let idx = ABSOLUTE_POS;
    let _half_n = n_fft / 2;
    
    let batch_id = idx / n_fft;
    let local_idx = idx % n_fft;
    
    let rev_local_idx = reverse_bits(local_idx, bits);
    
    if local_idx < rev_local_idx {
        let batch_offset = batch_id * n_fft;
        let idx1 = batch_offset + local_idx;
        let idx2 = batch_offset + rev_local_idx;
        
        let r1 = real[idx1];
        let i1 = imag[idx1];
        let r2 = real[idx2];
        let i2 = imag[idx2];
        
        real[idx1] = r2;
        imag[idx1] = i2;
        real[idx2] = r1;
        imag[idx2] = i1;
    }
}

#[cube(launch)]
pub fn fft_butterfly_kernel<F: Float>(
    real: &mut Tensor<F>,
    imag: &mut Tensor<F>,
    group_size: u32,
    n_fft: u32,
) {
    let idx = ABSOLUTE_POS;
    
    let half_n = n_fft / 2;
    
    let batch_id = idx / half_n;
    let local_idx = idx % half_n;
    
    let batch_offset = batch_id * n_fft;
    
    let half_len = group_size / 2;
    let i_base = (local_idx / half_len) * group_size;
    let j = local_idx % half_len;
    
    let idx1 = batch_offset + i_base + j;
    let idx2 = idx1 + half_len;
    
    let pi = 3.14159265359;
    let angle = F::new(-2.0) * F::new(pi) * F::cast_from(j) / F::cast_from(group_size);
    let w_real = F::cos(angle);
    let w_imag = F::sin(angle);
    
    let u_real = real[idx1];
    let u_imag = imag[idx1];
    
    let v_real_in = real[idx2];
    let v_imag_in = imag[idx2];
    
    let v_real = v_real_in * w_real - v_imag_in * w_imag;
    let v_imag = v_real_in * w_imag + v_imag_in * w_real;
    
    real[idx1] = u_real + v_real;
    imag[idx1] = u_imag + v_imag;
    
    real[idx2] = u_real - v_real;
    imag[idx2] = u_imag - v_imag;
}

pub trait FftBackend: Backend {
    fn fft_1d_batch_impl(
        real: FloatTensor<Self>,
        imag: FloatTensor<Self>,
        n_fft: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FftBackend for CubeBackend<R, F, I, BT> {
    fn fft_1d_batch_impl(
        real: FloatTensor<Self>,
        imag: FloatTensor<Self>,
        n_fft: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Ensure contiguous memory layout
        let real = into_contiguous(real);
        let imag = into_contiguous(imag);
        
        let total_elements = real.shape.num_elements();
        let num_batches = total_elements / n_fft;
        
        assert!(n_fft.is_power_of_two(), "FFT size must be power of 2");
        let bits = (n_fft as f32).log2() as u32;
        
        let client = &real.client;
        
        // 1. Bit Reversal
        let cube_dim = CubeDim::new_1d(256);
        let total_threads = total_elements;
        let cube_count = CubeCount::Static((total_threads as u32 + cube_dim.x - 1) / cube_dim.x, 1, 1);
        
        bit_reverse_kernel::launch::<F, R>(
            client,
            cube_count,
            cube_dim,
            real.as_tensor_arg(1),
            imag.as_tensor_arg(1),
            ScalarArg::new(n_fft as u32),
            ScalarArg::new(bits),
        ).unwrap();
        
        // 2. Butterfly Stages
        let mut group_size = 2;
        
        while group_size <= n_fft {
            let num_butterflies_per_fft = n_fft / 2;
            let total_butterflies = num_batches * num_butterflies_per_fft;
            
            let cube_count = CubeCount::Static((total_butterflies as u32 + cube_dim.x - 1) / cube_dim.x, 1, 1);
            
            fft_butterfly_kernel::launch::<F, R>(
                client,
                cube_count,
                cube_dim,
                real.as_tensor_arg(1),
                imag.as_tensor_arg(1),
                ScalarArg::new(group_size as u32),
                ScalarArg::new(n_fft as u32),
            ).unwrap();
            
            group_size *= 2;
        }
        
        (real, imag)
    }
}

impl FftBackend for NdArray<f32> {
    fn fft_1d_batch_impl(
        real: FloatTensor<Self>,
        imag: FloatTensor<Self>,
        n_fft: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let mut real_arc = match real {
            NdArrayTensor::F32(storage) => storage.into_owned(),
            _ => panic!("Expected F32 tensor"),
        };
        let mut imag_arc = match imag {
            NdArrayTensor::F32(storage) => storage.into_owned(),
            _ => panic!("Expected F32 tensor"),
        };
        
        let mut real_array = real_arc.into_owned();
        let mut imag_array = imag_arc.into_owned();
        
        let shape = real_array.shape().to_vec();
        let last_dim = shape.len() - 1;
        assert_eq!(shape[last_dim], n_fft);
        
        // Ensure contiguous
        if !real_array.is_standard_layout() {
            real_array = real_array.as_standard_layout().into_owned();
        }
        if !imag_array.is_standard_layout() {
            imag_array = imag_array.as_standard_layout().into_owned();
        }
        
        let r_slice = real_array.as_slice_mut().expect("FFT input must be contiguous");
        let i_slice = imag_array.as_slice_mut().expect("FFT input must be contiguous");
        
        // 1. Create Plan ONCE (Arc<dyn Fft> is Send+Sync)
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);
        let scratch_len = fft.get_inplace_scratch_len();

        // 2. Use for_each_init to reuse buffers per thread
        r_slice.par_chunks_mut(n_fft)
            .zip(i_slice.par_chunks_mut(n_fft))
            .for_each_init(
                || {
                    // Thread-local allocations
                    (
                        vec![Complex::zero(); scratch_len], 
                        vec![Complex::zero(); n_fft]
                    )
                },
                |(scratch, buffer), (r_chunk, i_chunk)| {
                    // Copy to Complex buffer
                    for (j, (r, i)) in r_chunk.iter().zip(i_chunk.iter()).enumerate() {
                        buffer[j] = Complex::new(*r, *i);
                    }
                    
                    // Process with reused scratch buffer
                    fft.process_with_scratch(buffer, scratch);
                    
                    // Copy back
                    for (j, val) in buffer.iter().enumerate() {
                        r_chunk[j] = val.re;
                        i_chunk[j] = val.im;
                    }
                }
            );
            
        (
            NdArrayTensor::from(real_array.into_shared()),
            NdArrayTensor::from(imag_array.into_shared())
        )
    }
}

