use cubecl::{cube, prelude::*};
use burn::tensor::{backend::Backend, ops::FloatTensor, TensorPrimitive};
use burn::tensor::Tensor as BurnTensor;
use burn_cubecl::{CubeBackend, CubeRuntime, FloatElement, IntElement, BoolElement, tensor::CubeTensor, kernel::into_contiguous};
use burn_ndarray::{NdArray, NdArrayTensor};
use rayon::prelude::*;

#[cube(launch)]
pub fn sobel_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    height: u32,
    width: u32,
) {
    let idx = ABSOLUTE_POS;
    
    let y = idx / width;
    let x = idx % width;
    
    if y >= 1 && y < height - 1 && x >= 1 && x < width - 1 {
        // Indices
        let idx_tl = (y - 1) * width + (x - 1);
        let idx_t  = (y - 1) * width + x;
        let idx_tr = (y - 1) * width + (x + 1);
        
        let idx_l  = y * width + (x - 1);
        let idx_r  = y * width + (x + 1);
        
        let idx_bl = (y + 1) * width + (x - 1);
        let idx_b  = (y + 1) * width + x;
        let idx_br = (y + 1) * width + (x + 1);
        
        // Read values
        let tl = input[idx_tl];
        let t  = input[idx_t];
        let tr = input[idx_tr];
        let l  = input[idx_l];
        let r  = input[idx_r];
        let bl = input[idx_bl];
        let b  = input[idx_b];
        let br = input[idx_br];
        
        // Sobel X
        // -1 0 1
        // -2 0 2
        // -1 0 1
        let gx = (tr + F::new(2.0) * r + br) - (tl + F::new(2.0) * l + bl);
        
        // Sobel Y
        // -1 -2 -1
        //  0  0  0
        //  1  2  1
        let gy = (bl + F::new(2.0) * b + br) - (tl + F::new(2.0) * t + tr);
        
        let mag = F::sqrt(gx * gx + gy * gy);
        
        output[idx] = mag;
    } else {
        output[idx] = F::new(0.0);
    }
}

#[cube(launch)]
pub fn temporal_diff_kernel<F: Float>(
    current: &Tensor<F>,
    prev: &Tensor<F>,
    prev_prev: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let idx = ABSOLUTE_POS;
    
    if idx < output.len() {
        let c = current[idx];
        let p = prev[idx];
        let pp = prev_prev[idx];
        
        // Motion Energy: |Current - Prev| + |Prev - PrevPrev|
        let diff1 = F::abs(c - p);
        let diff2 = F::abs(p - pp);
        
        output[idx] = diff1 + diff2;
    }
}

pub trait OpsBackend: Backend {
    fn sobel_impl(
        input: FloatTensor<Self>,
        height: usize,
        width: usize,
    ) -> FloatTensor<Self>;
    
    fn temporal_diff_impl(
        current: FloatTensor<Self>,
        prev: FloatTensor<Self>,
        prev_prev: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> OpsBackend for CubeBackend<R, F, I, BT> {
    fn sobel_impl(
        input: FloatTensor<Self>,
        height: usize,
        width: usize,
    ) -> FloatTensor<Self> {
        let input = into_contiguous(input);
        let num_elems = input.shape.num_elements();
        let size_bytes = num_elems * core::mem::size_of::<F>();
        
        let output_handle = input.client.empty(size_bytes);
        
        let output_tensor = CubeTensor::new(
            input.client.clone(),
            output_handle,
            input.shape.clone(),
            input.device.clone(),
            input.strides.clone(),
            F::dtype(),
        );
        
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static((num_elems as u32 + cube_dim.x - 1) / cube_dim.x, 1, 1);
        
        sobel_kernel::launch::<F, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(1),
            output_tensor.as_tensor_arg(1),
            ScalarArg::new(height as u32),
            ScalarArg::new(width as u32),
        ).unwrap();
        
        output_tensor
    }
    
    fn temporal_diff_impl(
        current: FloatTensor<Self>,
        prev: FloatTensor<Self>,
        prev_prev: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let current = into_contiguous(current);
        let prev = into_contiguous(prev);
        let prev_prev = into_contiguous(prev_prev);
        
        let num_elems = current.shape.num_elements();
        let size_bytes = num_elems * core::mem::size_of::<F>();
        
        let output_handle = current.client.empty(size_bytes);
        
        let output_tensor = CubeTensor::new(
            current.client.clone(),
            output_handle,
            current.shape.clone(),
            current.device.clone(),
            current.strides.clone(),
            F::dtype(),
        );
        
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static((num_elems as u32 + cube_dim.x - 1) / cube_dim.x, 1, 1);
        
        temporal_diff_kernel::launch::<F, R>(
            &current.client,
            cube_count,
            cube_dim,
            current.as_tensor_arg(1),
            prev.as_tensor_arg(1),
            prev_prev.as_tensor_arg(1),
            output_tensor.as_tensor_arg(1),
        ).unwrap();
        
        output_tensor
    }
}

pub fn compute_sobel<B: Backend + OpsBackend>(input: BurnTensor<B, 2>) -> BurnTensor<B, 2> {
    let dims = input.shape().dims;
    let height = dims[0];
    let width = dims[1];
    
    let input_prim = input.into_primitive();
    let input_t = match input_prim {
        TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    
    let out_t = B::sobel_impl(input_t, height, width);
    
    BurnTensor::from_primitive(TensorPrimitive::Float(out_t))
}

pub fn compute_temporal_diff<B: Backend + OpsBackend>(
    current: BurnTensor<B, 2>,
    prev: BurnTensor<B, 2>,
    prev_prev: BurnTensor<B, 2>,
) -> BurnTensor<B, 2> {
    let current_prim = current.into_primitive();
    let prev_prim = prev.into_primitive();
    let prev_prev_prim = prev_prev.into_primitive();
    
    let current_t = match current_prim { TensorPrimitive::Float(t) => t, _ => panic!("Expected float tensor") };
    let prev_t = match prev_prim { TensorPrimitive::Float(t) => t, _ => panic!("Expected float tensor") };
    let prev_prev_t = match prev_prev_prim { TensorPrimitive::Float(t) => t, _ => panic!("Expected float tensor") };
    
    let out_t = B::temporal_diff_impl(current_t, prev_t, prev_prev_t);
    
    BurnTensor::from_primitive(TensorPrimitive::Float(out_t))
}

impl OpsBackend for NdArray<f32> {
    fn sobel_impl(
        input: FloatTensor<Self>,
        height: usize,
        width: usize,
    ) -> FloatTensor<Self> {
        let input_arc = match input {
            NdArrayTensor::F32(storage) => storage.into_owned(),
            _ => panic!("Expected F32 tensor"),
        };
        
        let mut output_array = ndarray::Array2::<f32>::zeros((height, width));
        
        let input_slice = input_arc.as_slice().expect("Sobel input must be contiguous");
        let out_slice = output_array.as_slice_mut().unwrap();
        
        out_slice.par_chunks_mut(width).enumerate().for_each(|(y, row_out)| {
            if y == 0 || y >= height - 1 {
                return;
            }
            
            // Direct slice access to avoid ndarray overhead
            let prev_row = &input_slice[(y - 1) * width .. y * width];
            let curr_row = &input_slice[y * width .. (y + 1) * width];
            let next_row = &input_slice[(y + 1) * width .. (y + 2) * width];
            
            for x in 1..width-1 {
                let tl = prev_row[x-1];
                let t  = prev_row[x];
                let tr = prev_row[x+1];
                
                let l  = curr_row[x-1];
                let r  = curr_row[x+1];
                
                let bl = next_row[x-1];
                let b  = next_row[x];
                let br = next_row[x+1];
                
                let gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
                let gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
                
                row_out[x] = (gx * gx + gy * gy).sqrt();
            }
        });
        
        NdArrayTensor::from(output_array.into_dyn().into_shared())
    }
    
    fn temporal_diff_impl(
        current: FloatTensor<Self>,
        prev: FloatTensor<Self>,
        prev_prev: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let c_arc = match current { NdArrayTensor::F32(s) => s.into_owned(), _ => panic!() };
        let p_arc = match prev { NdArrayTensor::F32(s) => s.into_owned(), _ => panic!() };
        let pp_arc = match prev_prev { NdArrayTensor::F32(s) => s.into_owned(), _ => panic!() };
        
        let mut out_arr = ndarray::ArrayD::<f32>::zeros(c_arc.shape());
        
        let c_slice = c_arc.as_slice().unwrap();
        let p_slice = p_arc.as_slice().unwrap();
        let pp_slice = pp_arc.as_slice().unwrap();
        let out_slice = out_arr.as_slice_mut().unwrap();
        
        out_slice.par_iter_mut()
            .zip(c_slice.par_iter())
            .zip(p_slice.par_iter())
            .zip(pp_slice.par_iter())
            .for_each(|(((o, c), p), pp)| {
                let diff1 = (c - p).abs();
                let diff2 = (p - pp).abs();
                *o = diff1 + diff2;
            });
            
        NdArrayTensor::from(out_arr.into_shared())
    }
}


