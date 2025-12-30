mod fft_kernel;
mod cube_fft;
mod cube_ops;

use burn::tensor::{Tensor, backend::Backend, TensorPrimitive};
use burn::backend::wgpu::WgpuRuntime;
use burn_cubecl::CubeBackend;
use cube_fft::FftBackend;
use cube_ops::{OpsBackend, compute_sobel, pack_rgb};
use std::io::Write;
use nokhwa::{Camera, utils::{RequestedFormat, RequestedFormatType}, pixel_format::RgbFormat};
use minifb::{Window, WindowOptions, Key, ScaleMode};

// Type alias for our backend
type MyBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let generate_video = args.contains(&"--generate-video".to_string());

    let device = burn::backend::wgpu::WgpuDevice::default();
    println!("Initializing 2D FFT on GPU: {:?}", device);

    if generate_video {
        run_video_generation(&device);
    } else {
        run_realtime_camera(&device);
    }
}

fn run_realtime_camera(device: &burn::backend::wgpu::WgpuDevice) {
    println!("Starting Realtime Camera Mode...");
    
    // 2. Setup Window
    let width = 256;
    let height = 256;
    let window_width = width * 3; // Input, FFT, Sobel
    let window_height = height;
    
    let mut window = Window::new(
        "Realtime 2D FFT & Sobel - Burn GPU",
        window_width,
        window_height,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::AspectRatioStretch,
            ..WindowOptions::default()
        },
    ).unwrap_or_else(|e| {
        panic!("{}", e);
    });
    
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600))); // ~60 FPS
    let mut buffer: Vec<u32> = vec![0; window_width * window_height];

    // Cyclic Buffer (Size 3)
    let mut ring_buffer: Vec<Tensor<MyBackend, 2>> = Vec::with_capacity(3);
    let mut ring_idx = 0;

    // 1. Setup Camera with Retry Loop
    let index = nokhwa::utils::CameraIndex::Index(0);
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    
    loop { // Outer Reconnection Loop
        if !window.is_open() || window.is_key_down(Key::Escape) {
            break;
        }

        let mut camera = loop {
            if !window.is_open() || window.is_key_down(Key::Escape) {
                return;
            }
            
            println!("Attempting to connect to camera...");
            match Camera::new(index.clone(), requested) {
                Ok(mut cam) => {
                    match cam.open_stream() {
                        Ok(_) => {
                            println!("Camera connected!");
                            break cam;
                        }
                        Err(e) => {
                            eprintln!("Camera found but failed to open stream: {}. Retrying in 1s...", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Could not access camera: {}. Retrying in 1s...", e);
                }
            }
            
            // Update window to keep it alive/responsive (displaying black/waiting)
            window.update_with_buffer(&buffer, window_width, window_height).unwrap();
            std::thread::sleep(std::time::Duration::from_secs(1));
        };
        
        let cam_fmt = camera.camera_format();
        println!("Camera Format: {:?}", cam_fmt);
        
        println!("Press ESC to exit.");
        
        let mut frame_count = 0;
        let mut last_print = std::time::Instant::now();

        while window.is_open() && !window.is_key_down(Key::Escape) {
            let _start_frame = std::time::Instant::now();
            
            // Capture Frame
            let frame = match camera.frame() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to capture frame: {}. Reconnecting...", e);
                    break; // Break inner loop to trigger reconnection
                }
            };
            
            let decoded = match frame.decode_image::<RgbFormat>() {
                Ok(img) => img,
                Err(e) => {
                    eprintln!("Failed to decode image: {}", e);
                    continue; // Decoding error might be transient
                }
            };
            
            // Resize/Crop to 256x256 for FFT
            let resized = image::imageops::resize(&decoded, width as u32, height as u32, image::imageops::FilterType::Nearest);
            
            // Convert to Tensor Input (Grayscale for FFT)
            let mut input_floats = Vec::with_capacity(width * height);
            for pixel in resized.pixels() {
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;
                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                input_floats.push(gray / 255.0);
            }
            
            // Upload to GPU
            let tensor = Tensor::<MyBackend, 1>::from_floats(input_floats.as_slice(), device);
            let tensor_2d = tensor.reshape([height, width]);
            
            // Update Ring Buffer
            if ring_buffer.len() < 3 {
                ring_buffer.push(tensor_2d.clone());
            } else {
                ring_buffer[ring_idx] = tensor_2d.clone();
                ring_idx = (ring_idx + 1) % 3;
            }
            
            // Get frames for RGB Split (Current, Prev, PrevPrev)
            // If buffer not full, use current for all
            let (r_frame, g_frame, b_frame) = if ring_buffer.len() < 3 {
                (tensor_2d.clone(), tensor_2d.clone(), tensor_2d.clone())
            } else {
                // ring_idx points to the *oldest* frame (next to be overwritten), 
                // so (ring_idx - 1) is the newest.
                // We want: R=Newest, G=Prev, B=Oldest
                let idx_0 = (ring_idx + 2) % 3; // Newest (Current)
                let idx_1 = (ring_idx + 1) % 3; // Previous
                let idx_2 = ring_idx;           // Oldest
                
                (ring_buffer[idx_0].clone(), ring_buffer[idx_1].clone(), ring_buffer[idx_2].clone())
            };
            
            // Perform RGB Temporal Pack (GPU)
            let rgb_packed = pack_rgb(r_frame, g_frame, b_frame);
            
            // Perform 2D FFT
            let fft_result = compute_fft_2d(tensor_2d.clone());
            
            // Perform Sobel Edge Detection
            let sobel_result = compute_sobel(tensor_2d);
            
            // Download Results
            let fft_data = fft_result.to_data();
            let fft_vals = fft_data.as_slice::<f32>().unwrap();
            
            let sobel_data = sobel_result.to_data();
            let sobel_vals = sobel_data.as_slice::<f32>().unwrap();
            
            let rgb_data = rgb_packed.to_data();
            let rgb_vals = rgb_data.as_slice::<i32>().unwrap();
            
            // Visualization
            // Find max magnitude for normalization
            let mut max_mag = 0.0f32;
            let mut magnitudes = Vec::with_capacity(width * height);
            
            for j in 0..(width * height) {
                let r = fft_vals[j * 2];
                let im = fft_vals[j * 2 + 1];
                let mag = (r * r + im * im).sqrt();
                let log_mag = (1.0 + mag).ln();
                magnitudes.push(log_mag);
                if log_mag > max_mag {
                    max_mag = log_mag;
                }
            }
            
            if max_mag == 0.0 { max_mag = 1.0; }
            
            // Update Window Buffer
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    
                    // 1. Left: RGB Temporal Split (Ghosting Effect)
                    // We already packed it on GPU!
                    let color_rgb = rgb_vals[idx] as u32;
                    buffer[y * window_width + x] = color_rgb;
                    
                    // 2. Middle: FFT Magnitude (Shifted)
                    let shift_y = (y + height / 2) % height;
                    let shift_x = (x + width / 2) % width;
                    let mag_idx = shift_y * width + shift_x;
                    let mag = magnitudes[mag_idx];
                    
                    let val = ((mag / max_mag) * 255.0) as u32;
                    let color_fft = (val << 16) | (val << 8) | val;
                    
                    buffer[y * window_width + (x + width)] = color_fft;
                    
                    // 3. Right: Sobel Edge Detection
                    let sobel_val = sobel_vals[idx];
                    let val = (sobel_val * 255.0).clamp(0.0, 255.0) as u32;
                    // Greenish for edges
                    let color_sobel = (val << 8); 
                    
                    buffer[y * window_width + (x + width * 2)] = color_sobel;
                }
            }
            
            window.update_with_buffer(&buffer, window_width, window_height).unwrap();
            
            frame_count += 1;
            if frame_count % 60 == 0 {
                let elapsed = last_print.elapsed();
                println!("FPS: {:.2}", 60.0 / elapsed.as_secs_f64());
                last_print = std::time::Instant::now();
            }
        }
        
        // If we broke out of the inner loop but window is still open, we loop back to reconnect.
        // We can add a small delay here to avoid instant retry loops if something is weird.
        if window.is_open() && !window.is_key_down(Key::Escape) {
             println!("Connection lost. Restarting connection loop...");
             std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }
}

fn run_video_generation(device: &burn::backend::wgpu::WgpuDevice) {
    eprintln!("Generating test video frames...");
    let width = 256;
    let height = 256;
    let frames = 120;
    
    let mut video_data = Vec::new();
    for f in 0..frames {
        let frame = generate_frame(width, height, f);
        video_data.push(frame);
    }
    
    eprintln!("Generated {} frames of size {}x{}", frames, width, height);
    eprintln!("Starting real-time processing...");

    let mut stdout = std::io::stdout();
    
    for (_i, frame_data) in video_data.iter().enumerate() {
        // Upload to GPU
        let tensor = Tensor::<MyBackend, 1>::from_floats(frame_data.as_slice(), device);
        let tensor_2d = tensor.reshape([height, width]);
        
        // Perform 2D FFT
        let fft_result = compute_fft_2d(tensor_2d);
        
        // Visualization
        let fft_data = fft_result.to_data();
        let fft_vals = fft_data.as_slice::<f32>().unwrap(); // [H, W, 2] flattened
        
        let mut rgb_frame = Vec::with_capacity(width * height * 3 * 2); // Side by side
        
        // Find max magnitude for normalization
        let mut max_mag = 0.0f32;
        let mut magnitudes = Vec::with_capacity(width * height);
        
        for j in 0..(width * height) {
            let r = fft_vals[j * 2];
            let im = fft_vals[j * 2 + 1];
            let mag = (r * r + im * im).sqrt();
            let log_mag = (1.0 + mag).ln();
            magnitudes.push(log_mag);
            if log_mag > max_mag {
                max_mag = log_mag;
            }
        }
        
        if max_mag == 0.0 { max_mag = 1.0; }

        for y in 0..height {
            for x in 0..width {
                // Left: Input
                let input_val = frame_data[y * width + x];
                let pixel_val = (input_val * 255.0).clamp(0.0, 255.0) as u8;
                rgb_frame.push(pixel_val);
                rgb_frame.push(pixel_val);
                rgb_frame.push(pixel_val);
            }
            
            for x in 0..width {
                // Right: FFT Magnitude (Shifted)
                let shift_y = (y + height / 2) % height;
                let shift_x = (x + width / 2) % width;
                let mag_idx = shift_y * width + shift_x;
                let mag = magnitudes[mag_idx];
                
                let pixel_val = ((mag / max_mag) * 255.0).clamp(0.0, 255.0) as u8;
                
                rgb_frame.push(pixel_val);
                rgb_frame.push(pixel_val);
                rgb_frame.push(pixel_val);
            }
        }
        
        stdout.write_all(&rgb_frame).unwrap();
    }
    eprintln!("Video generation complete.");
}

fn generate_frame(width: usize, height: usize, frame_idx: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(width * height);
    let time = frame_idx as f32 * 0.1;
    for y in 0..height {
        for x in 0..width {
            // Moving circle pattern
            let cx = (width as f32 / 2.0) + (time.cos() * width as f32 / 4.0);
            let cy = (height as f32 / 2.0) + (time.sin() * height as f32 / 4.0);
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx*dx + dy*dy).sqrt();
            
            let val = if dist < 20.0 { 1.0 } else { 0.0 };
            data.push(val);
        }
    }
    data
}

fn compute_fft_2d<B: Backend + FftBackend>(input: Tensor<B, 2>) -> Tensor<B, 3> {
    let dims = input.shape().dims;
    let height = dims[0];
    let width = dims[1];
    
    // 1. FFT on Rows (last dim)
    // Input is [H, W]
    // We treat it as H batches of size W
    
    let real = input;
    let imag = Tensor::zeros_like(&real);
    
    let (real, imag) = run_fft_batch(real, imag, width);
    
    // 2. Transpose to [W, H]
    let real = real.transpose();
    let imag = imag.transpose();
    
    // 3. FFT on Columns (now rows)
    // We treat it as W batches of size H
    let (real, imag) = run_fft_batch(real, imag, height);
    
    // 4. Transpose back to [H, W]
    let real = real.transpose();
    let imag = imag.transpose();
    
    // Combine to [H, W, 2]
    let real = real.reshape([height, width, 1]);
    let imag = imag.reshape([height, width, 1]);
    
    Tensor::cat(vec![real, imag], 2)
}

fn run_fft_batch<B: Backend + FftBackend>(
    real: Tensor<B, 2>, 
    imag: Tensor<B, 2>, 
    n_fft: usize
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let dims = real.shape().dims;
    let _batch = dims[0];
    
    // Extract primitives
    let real_prim = real.into_primitive();
    let imag_prim = imag.into_primitive();
    
    let real_t = match real_prim {
        TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    
    let imag_t = match imag_prim {
        TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    };
    
    // Run Kernel
    let (real_out_t, imag_out_t) = B::fft_1d_batch_impl(real_t, imag_t, n_fft);
    
    // Wrap back
    let real_res: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(real_out_t));
    let imag_res: Tensor<B, 2> = Tensor::from_primitive(TensorPrimitive::Float(imag_out_t));
    
    (real_res, imag_res)
}
