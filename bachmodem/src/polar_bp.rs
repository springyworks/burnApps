/// Polar Codes with Belief Propagation (BP) Decoder on GPU
/// 
/// Implements a fully parallelizable BP decoder using Burn tensors.
/// This allows "crunching on GPU" as requested.

use burn::tensor::{Tensor, backend::Backend, ElementConversion};

pub struct PolarCodeBP {
    pub n: usize,
    pub k: usize,
    pub frozen_mask: Vec<bool>, // True if frozen (0)
}

impl PolarCodeBP {
    pub fn new(n: usize, k: usize) -> Self {
        // Reuse existing PolarCode logic for construction
        let pc = crate::polar::PolarCode::new(n, k);
        let mut frozen_mask = vec![false; n];
        for &idx in &pc.frozen_positions {
            frozen_mask[idx] = true;
        }
        
        Self { n, k, frozen_mask }
    }
    
    /// Decode using Belief Propagation on GPU
    /// llrs: [N] input LLRs
    /// iterations: Number of BP iterations (e.g., 20-50)
    pub fn decode_bp<B: Backend>(
        &self,
        device: &B::Device,
        llrs: &Tensor<B, 1>,
        iterations: usize,
    ) -> Tensor<B, 1> {
        let n = self.n;
        let stages = (n as f64).log2() as usize;
        
        // Initialize messages
        // Left messages (L): from channel to info bits
        // Right messages (R): from info bits to channel
        // Shape: [Stages + 1, N]
        
        // L[0] = Channel LLRs
        // R[Stages] = Infinity for frozen (0 known), 0 for info (unknown)
        
        let mut l = Tensor::<B, 2>::zeros([stages + 1, n], device);
        let mut r = Tensor::<B, 2>::zeros([stages + 1, n], device);
        
        // Set channel LLRs at stage 0
        // l[0, :] = llrs
        // We need to slice and assign. Burn doesn't support direct slice assignment easily yet.
        // We construct the tensor.
        
        // Actually, let's represent L and R as Vec<Tensor<B, 1>> for each stage
        // This avoids slice assignment issues.
        let mut l_stages: Vec<Tensor<B, 1>> = vec![Tensor::zeros([n], device); stages + 1];
        let mut r_stages: Vec<Tensor<B, 1>> = vec![Tensor::zeros([n], device); stages + 1];
        
        l_stages[0] = llrs.clone();
        
        // Initialize R at last stage (info bits)
        // Frozen bits: R = infinity (strong belief in 0)
        // Info bits: R = 0 (no prior)
        let mut r_init_data = vec![0.0f32; n];
        for i in 0..n {
            if self.frozen_mask[i] {
                r_init_data[i] = 1e9; // "Infinity"
            } else {
                r_init_data[i] = 0.0;
            }
        }
        r_stages[stages] = Tensor::from_floats(r_init_data.as_slice(), device);
        
        // BP Iterations
        for _iter in 0..iterations {
            // Left-to-Right Pass (Update L)
            for s in 0..stages {
                let step = 1 << (stages - 1 - s); // Stride size
                // In stage s, we process blocks of size 2*step
                // But the connections are fixed.
                // Let's use the standard indexing.
                // Stage s connects to s+1.
                // Node i at stage s connects to i and i+step at stage s+1 (if i < step in block)
                
                // Actually, let's use the recursive structure.
                // At stage s (0 to m-1), we have N/2 butterflies.
                // Stride = 2^s? No, depends on indexing.
                // Let's use the standard Peacan-matrix indexing or similar.
                
                // Standard Polar Factor Graph:
                // Stage 0 (Channel) -> ... -> Stage m (Bits)
                // At stage s (0..m-1), stride = 2^s.
                // Node i and i+stride are connected.
                
                let stride = 1 << s;
                
                // We need to perform operations on shifted tensors.
                // L_out_upper = f(L_in_upper, L_in_lower + R_out_lower)
                // L_out_lower = f(L_in_upper, R_out_upper) + L_in_lower
                // Wait, update rules:
                // L_{s+1, i} = f(L_{s, i}, L_{s, i+stride} + R_{s+1, i+stride})
                // L_{s+1, i+stride} = f(L_{s, i}, R_{s+1, i}) + L_{s, i+stride}
                // Where f(a, b) ≈ sign(a)sign(b)min(|a|, |b|)
                
                // We can implement this using tensor shifts (roll/slice)
                // But the stride changes.
                
                // Let's construct the permutation indices for this stage?
                // Or just use reshape?
                // Reshape [N] -> [N/2stride, 2, stride]
                // Then we can operate on the '2' dimension.
                
                let num_groups = n / (2 * stride);
                
                let l_in = l_stages[s].clone();
                let r_out = r_stages[s + 1].clone();
                
                // Reshape to separate upper and lower branches
                // [NumGroups, 2, Stride]
                let l_in_reshaped = l_in.reshape([num_groups, 2, stride]);
                let r_out_reshaped = r_out.reshape([num_groups, 2, stride]);
                
                // Slice
                // upper: index 0, lower: index 1
                let l_in_u = l_in_reshaped.clone().slice([0..num_groups, 0..1, 0..stride]).reshape([num_groups, stride]);
                let l_in_l = l_in_reshaped.clone().slice([0..num_groups, 1..2, 0..stride]).reshape([num_groups, stride]);
                
                let r_out_u = r_out_reshaped.clone().slice([0..num_groups, 0..1, 0..stride]).reshape([num_groups, stride]);
                let r_out_l = r_out_reshaped.clone().slice([0..num_groups, 1..2, 0..stride]).reshape([num_groups, stride]);
                
                // Update rules (Min-Sum approximation)
                // L_out_u = f(L_in_u, L_in_l + R_out_l)
                // L_out_l = f(L_in_u, R_out_u) + L_in_l
                
                let sum_lr = l_in_l.clone() + r_out_l;
                let l_out_u = min_sum(l_in_u.clone(), sum_lr);
                
                let sum_ur = min_sum(l_in_u, r_out_u);
                let l_out_l = sum_ur + l_in_l;
                
                // Combine back
                // Stack along dim 1: [NumGroups, 2, Stride]
                let l_out_stacked: Tensor<B, 3> = Tensor::stack(vec![l_out_u, l_out_l], 1);
                l_stages[s + 1] = l_out_stacked.reshape([n]);
            }
            
            // Right-to-Left Pass (Update R)
            for s in (0..stages).rev() {
                let stride = 1 << s;
                let num_groups = n / (2 * stride);
                
                let r_in = r_stages[s + 1].clone();
                let l_in = l_stages[s].clone();
                
                let r_in_reshaped = r_in.reshape([num_groups, 2, stride]);
                let l_in_reshaped = l_in.reshape([num_groups, 2, stride]);
                
                let r_in_u = r_in_reshaped.clone().slice([0..num_groups, 0..1, 0..stride]).reshape([num_groups, stride]);
                let r_in_l = r_in_reshaped.clone().slice([0..num_groups, 1..2, 0..stride]).reshape([num_groups, stride]);
                
                let l_in_u = l_in_reshaped.clone().slice([0..num_groups, 0..1, 0..stride]).reshape([num_groups, stride]);
                let l_in_l = l_in_reshaped.clone().slice([0..num_groups, 1..2, 0..stride]).reshape([num_groups, stride]);
                
                // Update rules
                // R_out_u = f(R_in_u, L_in_l + R_in_l)
                // R_out_l = f(R_in_u, L_in_u) + R_in_l
                
                let sum_lr = l_in_l + r_in_l.clone();
                let r_out_u = min_sum(r_in_u.clone(), sum_lr);
                
                let sum_ul = min_sum(r_in_u, l_in_u);
                let r_out_l = sum_ul + r_in_l;
                
                let r_out_stacked: Tensor<B, 3> = Tensor::stack(vec![r_out_u, r_out_l], 1);
                r_stages[s] = r_out_stacked.reshape([n]);
            }
        }
        
        // Final decision based on L at last stage + R at last stage (priors)
        // But R at last stage is just the priors we set.
        // So just look at L[stages] + R[stages]
        
        let final_llr = l_stages[stages].clone() + r_stages[stages].clone();
        
        // Hard decision: LLR < 0 => 1, LLR > 0 => 0
        // We return the LLRs, caller can threshold.
        final_llr
    }
}

/// Min-Sum approximation: f(a, b) ≈ sign(a)sign(b) min(|a|, |b|)
fn min_sum<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
    let sign_a = a.clone().sign();
    let sign_b = b.clone().sign();
    let abs_a = a.abs();
    let abs_b = b.abs();
    
    // min(x, y) = 0.5 * (x + y - |x - y|)
    // Burn might have element-wise min?
    // It does not have min(Tensor, Tensor) easily exposed in all backends, 
    // but we can use mask or the formula.
    // Let's use the formula for GPU efficiency (no branching).
    
    let min_abs = (abs_a.clone() + abs_b.clone() - (abs_a - abs_b).abs()) * 0.5;
    
    sign_a * sign_b * min_abs
}

/// Helper to sign: 1 if >=0, -1 if <0
trait SignTensor<B: Backend> {
    fn sign(self) -> Self;
}

impl<B: Backend> SignTensor<B> for Tensor<B, 2> {
    fn sign(self) -> Self {
        // x >= 0 -> 1, x < 0 -> -1
        // mask = x >= 0
        // res = mask * 1 + (!mask) * -1
        // res = mask * 2 - 1
        let mask = self.clone().greater_equal_elem(0.0).float();
        mask * 2.0 - 1.0
    }
}
