/// Final System Test: 5 Repetitions + Coherent Combining + Polar BP on GPU
/// 
/// Demonstrates the full stack:
/// 1. 5 Repetitions (Time Diversity)
/// 2. RAKE Receiver (Multipath Diversity)
/// 3. Soft Combining (Coherent LLR combining)
/// 4. Polar BP Decoder (GPU accelerated FEC)

use bachmodem::{
    write_wav, WattersonChannel,
    interleave, deinterleave, 
    PolarCode, PolarCodeBP, soft_bits_to_llrs, compute_soft_bits,
    RakeReceiver, TimeSlotConfig, generate_repetition_transmission,
    soft_combine_gpu,
    modulate_fhdpsk_with_flourishes, demodulate_fhdpsk_soft,
};
use burn::backend::Wgpu;
use burn::tensor::{Tensor, Distribution, ElementConversion};

type Backend = Wgpu;

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  BachModem FINAL SYSTEM TEST                              ║");
    println!("║  5 Reps + Coherent Combining + Polar BP (GPU)             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    let device = Default::default();
    
    // Configuration
    let message = "BachModem 73!";
    let snr_db = -25.0; // Very weak signal!
    let num_reps = 2; // Fast test: reduced from 5
    let use_fading = true;
    
    println!("Target: {} dB SNR with Watterson Fading", snr_db);
    println!("Message: \"{}\"", message);
    
    // 1. Encode
    println!("\n[TX] Encoding...");
    let data_bytes = message.as_bytes();
    let mut data_bits = Vec::new();
    for &byte in data_bytes {
        for i in (0..8).rev() {
            data_bits.push((byte >> i) & 1);
        }
    }
    
    // Polar Encode (N=256, K=128)
    // Pad to 128 bits
    let mut padded_bits = data_bits.clone();
    padded_bits.resize(128, 0);
    
    let polar = PolarCode::new(256, 128);
    let encoded_bits = polar.encode(&padded_bits);
    
    // Interleave
    let interleaved_bits = interleave(&encoded_bits, 16);
    
    // Pack to bytes for modulation
    let mut tx_bytes = Vec::new();
    for chunk in interleaved_bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        tx_bytes.push(byte);
    }
    
    // 2. Generate Repetitions
    println!("[TX] Generating {} repetitions...", num_reps);
    let config = TimeSlotConfig::new(tx_bytes.len(), num_reps, 0.5); // 0.5s gap (reduced from 2s)
    let clean_signal = generate_repetition_transmission::<Backend>(&device, &tx_bytes, &config);
    
    // 3. Channel Simulation
    println!("[Channel] Applying Watterson fading & Noise ({} dB)...", snr_db);
    let signal_power: f32 = clean_signal.clone().powf_scalar(2.0).mean().into_scalar().elem();
    let snr_linear = 10f32.powf(snr_db / 10.0);
    let noise_std = (signal_power / snr_linear).sqrt();
    
    let faded_signal = if use_fading {
        let channel = WattersonChannel::moderate();
        channel.apply::<Backend>(&device, &clean_signal)
    } else {
        clean_signal.clone()
    };
    
    let noise = Tensor::<Backend, 1>::random(
        faded_signal.shape(),
        Distribution::Normal(0.0, noise_std as f64),
        &device,
    );
    
    let rx_signal = faded_signal + noise;
    
    // 4. Receiver Processing
    println!("\n[RX] Processing...");
    
    // Search for the first repetition to establish timing
    // The decoder must find the start itself, as requested.
    println!("[RX] Searching for first repetition signal...");
    
    // Search in the first 40 seconds (320000 samples) to cover the 30s preamble
    let search_window_len = 320000.min(rx_signal.dims()[0]);
    let search_window = rx_signal.clone().slice([0..search_window_len]);
    
    let time_offset_samples = match bachmodem::modulation::synchronize_signal::<Backend>(&device, &search_window) {
        Some(pos) => {
            println!("  [RX] SYNC: Found first repetition at sample {} ({:.3}s)", pos, pos as f32 / 8000.0);
            pos
        },
        None => {
            println!("  [RX] SYNC: Failed to find first repetition! Aborting.");
            return;
        }
    };

    let mut all_llrs: Vec<Tensor<Backend, 1>> = Vec::new();
    
    for i in 0..num_reps {
        // Calculate expected position based on the first detected repetition
        // Slot spacing is fixed by the protocol (TimeSlotConfig)
        let slot_duration_samples = (config.transmission_duration * 8000.0) as usize;
        let gap_samples = (config.listening_gap * 8000.0) as usize;
        let stride = slot_duration_samples + gap_samples;
        
        let expected_start = time_offset_samples + i * stride;
        
        // Define a window around the expected start for the RAKE receiver
        // We give it some margin (+/- 2000 samples) to handle drift or jitter
        let margin = 2000;
        let window_start = expected_start.saturating_sub(margin);
        let window_end = (expected_start + slot_duration_samples + margin).min(rx_signal.dims()[0]);
        
        if window_start >= rx_signal.dims()[0] { break; }
        
        println!("  [RX] Processing Repetition {}/{} (Window: {}..{})", i+1, num_reps, window_start, window_end);
        
        let slot_signal = rx_signal.clone().slice([window_start..window_end]);
        
        // RAKE Receiver
        let mut rake = RakeReceiver::new(3, 200);
        // Use preamble as reference for RAKE
        let preamble = bachmodem::wavelet::generate_bach_preamble::<Backend>(&device);
        let processed_signal = rake.process::<Backend>(&device, &slot_signal, &preamble);
        
        // Demodulate to Soft Bits (LLRs) directly on GPU
        let llrs = demodulate_fhdpsk_soft::<Backend>(&device, &processed_signal, true, 64);
        
        let num_llrs = llrs.dims()[0];
        if num_llrs >= 256 {
            // Truncate to 256 bits
            let llrs_trunc = llrs.slice([0..256]);
            
            // Deinterleave LLRs
            // We need to deinterleave the LLRs
            // Since deinterleave works on u8, let's map LLRs to indices, deinterleave, map back?
            // Or just implement deinterleave for Vec<f32>
            // Wait, we have the LLRs on GPU.
            // We need to shuffle them.
            // Let's pull to CPU, shuffle, push back (fast enough for 256 floats)
            // Or implement shuffle on GPU.
            
            let llr_data = llrs_trunc.to_data();
            let llr_slice = llr_data.as_slice::<f32>().unwrap();
            
            let mut deint_llrs = vec![0.0f32; 256];
            let num_cols = 16;
            let num_rows = 16; // 256/16
            for j in 0..256 {
                let col = j / num_rows;
                let row = j % num_rows;
                let input_idx = row * num_cols + col;
                if input_idx < 256 {
                    deint_llrs[input_idx] = llr_slice[j];
                }
            }
            
            all_llrs.push(Tensor::from_floats(deint_llrs.as_slice(), &device));
            println!("  Rep {}: Decoded {} soft bits", i+1, num_llrs);
        } else {
            println!("  Rep {}: Failed decode (got {} bits)", i+1, num_llrs);
            // Push zeros (neutral LLRs)
            all_llrs.push(Tensor::zeros([256], &device));
        }
    }
    
    if all_llrs.is_empty() {
        println!("No data decoded.");
        return;
    }
    
    // 5. Soft Combine on GPU
    println!("[RX] Soft Combining {} copies on GPU...", all_llrs.len());
    let llr_stack = Tensor::stack(all_llrs.clone(), 0); // [NumReps, 256]
    
    // Adjust weights to match actual count
    let actual_reps = all_llrs.len();
    let weights = Tensor::ones([actual_reps], &device); 
    let combined_llrs = soft_combine_gpu(&llr_stack, &weights);
    
    // 6. Polar BP Decode on GPU
    println!("[RX] Polar BP Decoding on GPU...");
    let polar_bp = PolarCodeBP::new(256, 128);
    let decoded_llrs = polar_bp.decode_bp(&device, &combined_llrs, 20); // 20 iterations
    
    // Hard decision
    let decoded_data = decoded_llrs.to_data();
    let decoded_slice = decoded_data.as_slice::<f32>().unwrap();
    let mut final_bits = Vec::new();
    
    // Extract info bits
    for &pos in &polar.info_positions {
        let val = decoded_slice[pos];
        final_bits.push(if val < 0.0 { 1 } else { 0 });
    }
    
    // 7. Verify
    let mut final_bytes = Vec::new();
    for chunk in final_bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        final_bytes.push(byte);
    }
    
    println!("Decoded Bytes: {:?}", final_bytes);
    
    // Trim padding
    let len = message.len();
    if final_bytes.len() > len {
        final_bytes.truncate(len);
    }
    
    match String::from_utf8(final_bytes.clone()) {
        Ok(msg) => {
            println!("\n[Result] Decoded Message: \"{}\"", msg);
            if msg == message {
                println!("✨ SUCCESS! Perfect decode at {} dB", snr_db);
            } else {
                println!("⚠ Errors detected.");
            }
        }
        Err(_) => println!("\n[Result] Decoded invalid UTF-8"),
    }
}
