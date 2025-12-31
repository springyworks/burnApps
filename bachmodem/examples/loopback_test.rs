use bachmodem::{
    modulate_fhdpsk_with_flourishes, demodulate_fhdpsk_soft,
    PolarCode, PolarCodeBP,
    interleave, deinterleave_gpu,
};
use burn::tensor::Tensor;
use burn::backend::wgpu::{CubeBackend, WgpuRuntime};

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

fn main() {
    let device = Default::default();
    
    // Test message
    let message = b"BachModem Test";
    println!("Original: {:?}", String::from_utf8_lossy(message));
    
    // Encode with Polar
    let mut bits = Vec::new();
    for &byte in message {
        for i in (0..8).rev() {
            bits.push((byte >> i) & 1);
        }
    }
    // Pad to 128 bits
    bits.resize(128, 0);
    
    let polar = PolarCode::new(256, 128);
    let encoded_bits = polar.encode(&bits);
    println!("Polar encoded: {} bits", encoded_bits.len());
    
    print!("First 16 encoded bits: ");
    for i in 0..16 {
        print!("{}", encoded_bits[i]);
    }
    println!();
    
    // Interleave
    let interleaved = interleave(&encoded_bits, 16);
    
    print!("First 16 interleaved bits: ");
    for i in 0..16 {
        print!("{}", interleaved[i]);
    }
    println!();
    
    // Pack bits into bytes
    let mut tx_bytes = Vec::new();
    for chunk in interleaved.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        tx_bytes.push(byte);
    }
    
    // Modulate
    let signal = modulate_fhdpsk_with_flourishes::<Backend>(
        &device,
        &tx_bytes,
        true,
        32,
    );
    
    println!("Signal length: {} samples", signal.dims()[0]);
    
    // Add minimal noise (high SNR)
    let noise = Tensor::random(signal.shape(), burn::tensor::Distribution::Normal(0.0, 0.01), &device);
    let noisy_signal = signal + noise;
    
    // Demodulate
    let llrs = demodulate_fhdpsk_soft::<Backend>(&device, &noisy_signal, true, 32);
    
    println!("Demod LLRs: {} values", llrs.dims()[0]);
    
    // Print first 16 LLRs
    let llr_data = llrs.to_data();
    let llr_slice = llr_data.as_slice::<f32>().unwrap();
    print!("First 16 LLRs: ");
    for i in 0..16.min(llr_slice.len()) {
        print!("{:.2} ", llr_slice[i]);
    }
    println!();
    
    // Deinterleave
    let llr_vec: Vec<f32> = llr_slice.to_vec();
    let deinterleaved_llrs = deinterleave_gpu::<Backend>(&device, &Tensor::from_floats(llr_vec.as_slice(), &device), 16);
    
    let deint_data = deinterleaved_llrs.to_data();
    let deint_slice = deint_data.as_slice::<f32>().unwrap();
    
    print!("First 16 deinterleaved LLRs: ");
    for i in 0..16 {
        print!("{:.2} ", deint_slice[i]);
    }
    println!();
    
    // Decode with Polar BP
    println!("Decoding with Polar BP...");
    let polar_bp = PolarCodeBP::new(256, 128);
    let decoded_llrs_tensor = polar_bp.decode_bp(&device, &deinterleaved_llrs, 50);
    let decoded_llrs_data = decoded_llrs_tensor.to_data();
    let decoded_llrs = decoded_llrs_data.as_slice::<f32>().unwrap();
    
    // Extract info bits
    let mut decoded_bits = Vec::new();
    for &pos in &polar.info_positions {
        let llr = decoded_llrs[pos];
        // LLR > 0 => 0, LLR < 0 => 1
        decoded_bits.push(if llr < 0.0 { 1 } else { 0 });
    }
    
    print!("First 16 decoded bits: ");
    for i in 0..16 {
        print!("{}", decoded_bits[i]);
    }
    println!();
    
    // Pack into bytes
    let mut decoded_bytes = Vec::new();
    for chunk in decoded_bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        decoded_bytes.push(byte);
    }
    
    if decoded_bytes.len() > message.len() {
        decoded_bytes.truncate(message.len());
    }
    
    println!("Decoded: {:?}", String::from_utf8_lossy(&decoded_bytes));
    
    // Check errors
    let mut errors = 0;
    for (i, (&orig, &dec)) in message.iter().zip(decoded_bytes.iter()).enumerate() {
        if orig != dec {
            println!("  Error at byte {}: expected 0x{:02x} got 0x{:02x}", i, orig, dec);
            errors += 1;
        }
    }
    
    if errors == 0 {
        println!("✓ Perfect decode!");
    } else {
        println!("✗ {} byte errors", errors);
    }
}
