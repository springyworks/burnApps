/// GPU-based deinterleaving operations
/// 
/// Keeps data on GPU instead of downloading to CPU for deinterleaving

use burn::tensor::{Tensor, backend::Backend};

/// Deinterleave LLRs on GPU using gather operation
/// 
/// Input: [N] interleaved LLRs
/// Output: [N] deinterleaved LLRs
/// 
/// Interleaving writes column-wise, reads row-wise (16x16 grid)
/// Deinterleaving reverses this: write row-wise, read column-wise
pub fn deinterleave_gpu<B: Backend>(
    device: &B::Device,
    interleaved: &Tensor<B, 1>,
    num_cols: usize,
) -> Tensor<B, 1> {
    let n = interleaved.dims()[0];
    let num_rows = n / num_cols;
    
    if num_rows * num_cols != n {
        // Not evenly divisible, return as-is
        return interleaved.clone();
    }
    
    // Reshape to matrix: [Rows, Cols]
    let matrix = interleaved.clone().reshape([num_rows, num_cols]);
    
    // Transpose: [Cols, Rows]
    let transposed = matrix.swap_dims(0, 1);
    
    // Flatten back: [N]
    transposed.reshape([n])
}

/// Interleave LLRs on GPU (for encoding)
pub fn interleave_gpu<B: Backend>(
    device: &B::Device,
    data: &Tensor<B, 1>,
    num_cols: usize,
) -> Tensor<B, 1> {
    let n = data.dims()[0];
    let num_rows = n / num_cols;
    
    if num_rows * num_cols != n {
        return data.clone();
    }
    
    // Reshape: [Rows, Cols]
    let matrix = data.clone().reshape([num_rows, num_cols]);
    
    // Transpose: [Cols, Rows]
    let transposed = matrix.swap_dims(0, 1);
    
    // Flatten: [N]
    transposed.reshape([n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use crate::gpu_test_utils::validate_roundtrip;
    
    type TestBackend = Wgpu;
    
    #[test]
    fn test_deinterleave_gpu() {
        let device = Default::default();
        
        // Create test data: 0, 1, 2, ..., 15
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);
        
        // Validate round-trip using GPU-only operations
        validate_roundtrip(
            &tensor,
            |t| interleave_gpu::<TestBackend>(&device, t, 4),
            |t| deinterleave_gpu::<TestBackend>(&device, t, 4),
            1e-5,
            "Interleave/Deinterleave"
        );
        
        println!("GPU deinterleave test passed!");
    }
}
