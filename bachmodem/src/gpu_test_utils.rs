/// GPU-native testing utilities
/// 
/// Validation functions that keep computation on GPU to avoid
/// synchronization points in tests.

use burn::tensor::{Tensor, backend::Backend, ElementConversion};

/// Assert two tensors are approximately equal without forcing CPU sync
/// 
/// This uses GPU operations only: element-wise difference, abs, max
/// Only syncs at the very end for the assertion itself.
pub fn assert_approx_eq_gpu<B: Backend>(
    a: &Tensor<B, 1>,
    b: &Tensor<B, 1>,
    epsilon: f32,
    msg: &str,
) {
    let diff = (a.clone() - b.clone()).abs();
    let max_diff: f32 = diff.max().into_scalar().elem();
    
    assert!(
        max_diff < epsilon,
        "{}: max difference {:.6} >= epsilon {:.6}",
        msg,
        max_diff,
        epsilon
    );
}

/// Assert tensor approximately equals a constant value
pub fn assert_approx_eq_scalar<B: Backend>(
    tensor: &Tensor<B, 1>,
    expected: f32,
    epsilon: f32,
    msg: &str,
) {
    let expected_tensor = Tensor::ones_like(tensor) * expected;
    assert_approx_eq_gpu(tensor, &expected_tensor, epsilon, msg);
}

/// Validate round-trip properties on GPU
/// 
/// Tests that function(inverse_function(x)) ≈ x without intermediate syncs
pub fn validate_roundtrip<B: Backend, F, G>(
    original: &Tensor<B, 1>,
    forward_fn: F,
    inverse_fn: G,
    epsilon: f32,
    name: &str,
) where
    F: Fn(&Tensor<B, 1>) -> Tensor<B, 1>,
    G: Fn(&Tensor<B, 1>) -> Tensor<B, 1>,
{
    let transformed = forward_fn(original);
    let recovered = inverse_fn(&transformed);
    
    assert_approx_eq_gpu(
        original,
        &recovered,
        epsilon,
        &format!("{} round-trip failed", name)
    );
}

/// Check tensor is normalized (mean ≈ 0, std ≈ 1) on GPU
pub fn assert_normalized<B: Backend>(
    tensor: &Tensor<B, 1>,
    epsilon: f32,
    msg: &str,
) {
    let mean = tensor.clone().mean();
    let mean_val: f32 = mean.clone().into_scalar().elem();
    
    let variance = (tensor.clone() - mean).powf_scalar(2.0).mean();
    let std_val: f32 = variance.sqrt().into_scalar().elem();
    
    assert!(
        mean_val.abs() < epsilon,
        "{}: mean {:.6} not near zero",
        msg,
        mean_val
    );
    
    assert!(
        (std_val - 1.0).abs() < epsilon,
        "{}: std {:.6} not near 1.0",
        msg,
        std_val
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    type TestBackend = Wgpu;
    
    #[test]
    fn test_approx_eq() {
        let device = Default::default();
        
        let a = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let b = Tensor::<TestBackend, 1>::from_floats([1.001, 2.001, 2.999], &device);
        
        assert_approx_eq_gpu(&a, &b, 0.01, "should pass");
    }
    
    #[test]
    #[should_panic]
    fn test_approx_eq_fail() {
        let device = Default::default();
        
        let a = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let b = Tensor::<TestBackend, 1>::from_floats([1.0, 2.1, 3.0], &device);
        
        assert_approx_eq_gpu(&a, &b, 0.01, "should fail");
    }
}
