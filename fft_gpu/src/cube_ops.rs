use cubecl::{cube, prelude::*};
use burn::tensor::{backend::Backend, ops::{FloatTensor, IntTensor}, TensorPrimitive};
use burn::tensor::Tensor as BurnTensor;
use burn_cubecl::{CubeBackend, CubeRuntime, FloatElement, IntElement, BoolElement, tensor::CubeTensor, kernel::into_contiguous};

#[cube(launch)]
pub fn pack_rgb_kernel<F: Float, I: Int>(
    r: &Tensor<F>,
    g: &Tensor<F>,
    b: &Tensor<F>,
    output: &mut Tensor<I>,
    num_elems: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < num_elems {
        let r_val = r[idx];
        let g_val = g[idx];
        let b_val = b[idx];
        
        // Clamp and scale
        let r_clamped = F::clamp(r_val, F::new(0.0), F::new(1.0));
        let g_clamped = F::clamp(g_val, F::new(0.0), F::new(1.0));
        let b_clamped = F::clamp(b_val, F::new(0.0), F::new(1.0));
        
        let r_int = u32::cast_from(r_clamped * F::new(255.0));
        let g_int = u32::cast_from(g_clamped * F::new(255.0));
        let b_int = u32::cast_from(b_clamped * F::new(255.0));
        
        let packed = (r_int << 16) | (g_int << 8) | b_int;
        output[idx] = I::cast_from(packed);
    }
}

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

pub trait OpsBackend: Backend {
    fn sobel_impl(
        input: FloatTensor<Self>,
        height: usize,
        width: usize,
    ) -> FloatTensor<Self>;

    fn pack_rgb_impl(
        r: FloatTensor<Self>,
        g: FloatTensor<Self>,
        b: FloatTensor<Self>,
    ) -> IntTensor<Self>;
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> OpsBackend for CubeBackend<R, F, I, BT> {
    fn pack_rgb_impl(
        r: FloatTensor<Self>,
        g: FloatTensor<Self>,
        b: FloatTensor<Self>,
    ) -> IntTensor<Self> {
        let r = into_contiguous(r);
        let g = into_contiguous(g);
        let b = into_contiguous(b);
        
        let num_elems = r.shape.num_elements();
        let size_bytes = num_elems * core::mem::size_of::<I>();
        
        let output_handle = r.client.empty(size_bytes);
        
        let output_tensor = CubeTensor::new(
            r.client.clone(),
            output_handle,
            r.shape.clone(),
            r.device.clone(),
            r.strides.clone(),
            I::dtype(),
        );
        
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::Static((num_elems as u32 + cube_dim.x - 1) / cube_dim.x, 1, 1);
        
        pack_rgb_kernel::launch::<F, I, R>(
            &r.client,
            cube_count,
            cube_dim,
            r.as_tensor_arg(1),
            g.as_tensor_arg(1),
            b.as_tensor_arg(1),
            output_tensor.as_tensor_arg(1),
            ScalarArg::new(num_elems as u32),
        ).unwrap();
        
        output_tensor
    }

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

pub fn pack_rgb<B: Backend + OpsBackend>(
    r: BurnTensor<B, 2>,
    g: BurnTensor<B, 2>,
    b: BurnTensor<B, 2>,
) -> BurnTensor<B, 2, burn::tensor::Int> {
    let r_prim = r.into_primitive();
    let g_prim = g.into_primitive();
    let b_prim = b.into_primitive();
    
    let r_t = match r_prim { TensorPrimitive::Float(t) => t, _ => panic!("Expected float tensor") };
    let g_t = match g_prim { TensorPrimitive::Float(t) => t, _ => panic!("Expected float tensor") };
    let b_t = match b_prim { TensorPrimitive::Float(t) => t, _ => panic!("Expected float tensor") };
    
    let out_t = B::pack_rgb_impl(r_t, g_t, b_t);
    
    BurnTensor::from_primitive(out_t)
}
