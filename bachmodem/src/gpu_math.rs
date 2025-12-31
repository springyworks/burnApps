/// GPU-native mathematical functions
/// 
/// Implements trigonometric and transcendental functions that stay on GPU
/// to avoid synchronization points in hot paths.

use burn::tensor::{Tensor, backend::Backend};

/// Fast arctan approximation on GPU
/// 
/// atan(x) ≈ (π/4)x + 0.273 * x * (1 - abs(x))
fn fast_atan<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let pi_over_4 = std::f32::consts::FRAC_PI_4;
    
    let term1 = x.clone().mul_scalar(pi_over_4);
    let abs_x = x.clone().abs();
    let term2 = x.mul_scalar(0.273).mul(abs_x.neg().add_scalar(1.0));
    
    term1.add(term2)
}

/// Simple and fast atan2 using fast_atan approximation
/// 
/// **NO SYNC POINT**: Pure GPU computation  
/// Best performance for typical demodulation use case
pub fn atan2_fast_gpu<B: Backend>(
    y: Tensor<B, 1>,
    x: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let pi = std::f32::consts::PI;
    let half_pi = std::f32::consts::FRAC_PI_2;
    
    // Safe division: clamp x away from zero
    let x_safe = x.clone().abs().clamp_min(1e-10);
    let x_with_sign = x.clone().sign().mul(x_safe);
    
    // Compute ratio z = y/x
    let z = y.clone().div(x_with_sign);
    
    // Fast atan approximation
    let atan_z = fast_atan(z);
    
    // Quadrant correction based on sign of x
    let x_negative = x.clone().lower(Tensor::zeros_like(&x));
    let y_negative = y.clone().lower(Tensor::zeros_like(&y));
    
    // Convert bool masks to float (1.0 or 0.0)
    let x_neg_float = x_negative.float();
    let y_neg_float = y_negative.float();
    
    // Correction: add π if x<0 and y>=0, subtract π if x<0 and y<0
    let pi_tensor = Tensor::ones_like(&atan_z).mul_scalar(pi);
    let correction = pi_tensor.clone()
        .mul(x_neg_float.clone())
        .mul(Tensor::ones_like(&y_neg_float).sub(y_neg_float.clone()))
        .sub(pi_tensor.mul(x_neg_float).mul(y_neg_float));
    
    atan_z.add(correction)
}



#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::ElementConversion;
    
    type TestBackend = Wgpu;
    
    #[test]
    fn test_atan2_fast_gpu() {
        let device = Default::default();
        
        // Test basic cases
        let y = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0, 0.0, 1.0], &device);
        let x = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0, 1.0, -1.0], &device);
        
        let result = atan2_fast_gpu(y, x);
        let values: Vec<f32> = result.into_data().to_vec().unwrap();
        
        println!("atan2_fast_gpu results: {:?}", values);
        
        // Check approximate values
        assert!((values[0] - std::f32::consts::FRAC_PI_4).abs() < 0.1);
        assert!((values[1] - std::f32::consts::FRAC_PI_2).abs() < 0.1);
        assert!(values[2].abs() < 0.1);
    }
}
