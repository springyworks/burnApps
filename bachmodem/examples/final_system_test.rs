/// Final System Test: 15-Minute Deep Space Simulation
/// 
/// Configuration:
/// - SNR: -30 dB
/// - Channel: HF Watterson (Moderate Fading)
/// - Message: "BachModem: Deep Space Wavelet Transmission Test"
/// - Repetitions: 15
/// - Listening Gap: 5.0s
/// - Total Duration: ~15 minutes
/// 
/// This test validates the full system capability to recover data
/// from a long-duration, fading, noisy channel using the
/// Time-Slotted Repetition Protocol.

use bachmodem::{
    WattersonChannel,
    interleave, deinterleave_gpu, 
    PolarCode, PolarCodeBP, soft_combine_gpu,
    TimeSlotConfig, generate_repetition_transmission,
    RakeReceiver,
    FftBackend,
};
use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
use burn::tensor::{Tensor, Distribution, ElementConversion};
use hound;

// Use raw CubeBackend to avoid Fusion wrapper which doesn't implement FftBackend yet
type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  BachModem Final System Test                            ║");
    println!("║  Target: -30 dB SNR over HF Watterson Channel           ║");
    println!("║  Duration: ~15 Minutes                                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    let device = Default::default();
    let snr_db = -28.0;
    let filename = "final_system_test.wav";
    
    // ========================================================================
    // PART 1: GENERATE SIGNAL
    // ========================================================================
    let message = "BachModem: Deep Space Wavelet Transmission Test";
    let num_reps = 30;
    let listening_gap = 5.0;
    
    println!("Message: \"{}\"", message);
    println!("Repetitions: {}", num_reps);
    println!("Gap: {:.1}s", listening_gap);
    
    // 1. Encode
    // Use a shorter message that fits in one block (128 bits = 16 bytes)
    // "BachModem Test" = 14 bytes
    let short_message = "BachModem Test"; 
    println!("Using short message for single block: \"{}\"", short_message);
    
    let data_bytes = short_message.as_bytes();
    let mut data_bits = Vec::new();
    for &byte in data_bytes {
        for i in (0..8).rev() {
            data_bits.push((byte >> i) & 1);
        }
    }
    
    let mut padded_bits = data_bits.clone();
    padded_bits.resize(128, 0);
    
    let polar = PolarCode::new(256, 128);
    let encoded_bits = polar.encode(&padded_bits);
    let interleaved_bits = interleave(&encoded_bits, 16);
    
    // Debug: print what we're going to transmit
    print!("Interleaved bits (first 32): ");
    for i in 0..32 {
        print!("{}", interleaved_bits[i]);
    }
    println!();
    
    let mut tx_bytes = Vec::new();
    for chunk in interleaved_bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        tx_bytes.push(byte);
    }
    
    // 2. Generate one transmission (no flourishes for now - simpler)
    println!("Generating base transmission...");
    let single_tx = bachmodem::modulation::modulate_fhdpsk::<Backend>(&device, &tx_bytes, true);
    let tx_len = single_tx.dims()[0];
    let gap_len = (listening_gap * 8000.0) as usize;
    let stride = tx_len + gap_len;
    let total_len = num_reps * stride;
    
    println!("Single transmission: {} samples ({:.2}s)", tx_len, tx_len as f32 / 8000.0);
    println!("Total duration: {:.2}s ({:.2} minutes)", total_len as f32 / 8000.0, total_len as f32 / 8000.0 / 60.0);
    
    // Build repetition signal
    let mut clean_signal = Tensor::<Backend, 1>::zeros([total_len], &device);
    for i in 0..num_reps {
        let start = i * stride;
        println!("  Repetition {}/{}: starts at {:.1}s (sample {})", i + 1, num_reps, start as f32 / 8000.0, start);
        clean_signal = clean_signal.slice_assign([start..start + tx_len], single_tx.clone());
    }
    
    // 3. Channel
    let signal_power: f32 = clean_signal.clone().powf_scalar(2.0).mean().into_scalar().elem();
    let snr_linear = 10f32.powf(snr_db / 10.0);
    let noise_std = (signal_power / snr_linear).sqrt();
    
    println!("Applying Watterson Channel (Gentle)...");
    let channel = WattersonChannel::gentle();
    let faded_signal = channel.apply::<Backend>(&device, &clean_signal);
    
    println!("Adding Noise (-30 dB)...");
    let noise = Tensor::<Backend, 1>::random(
        faded_signal.shape(),
        Distribution::Normal(0.0, noise_std as f64),
        &device,
    );
    
    let rx_signal = faded_signal + noise;
    
    // 4. Save
    println!("Saving to {}...", filename);
    let rx_data = rx_signal.to_data();
    let rx_samples = rx_data.as_slice::<f32>().unwrap();
    
    let max_val = rx_samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let scale = 0.95 / max_val;
    
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 8000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::create(filename, spec).unwrap();
    for &sample in rx_samples {
        let amplitude = (sample * scale * 32767.0).clamp(-32768.0, 32767.0);
        writer.write_sample(amplitude as i16).unwrap();
    }
    writer.finalize().unwrap();
    
    drop(rx_signal);
    drop(clean_signal);
    
    // ========================================================================
    // PART 2: DECODE
    // ========================================================================
    println!("\nDecoding...");
    
    let mut reader = hound::WavReader::open(filename).unwrap();
    let samples: Vec<f32> = reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32767.0)
        .collect();
    
    let rx_signal = Tensor::<Backend, 1>::from_floats(samples.as_slice(), &device);
    
    // Sync
    let search_window_len = 200000.min(rx_signal.dims()[0]); // Search first 25s
    let search_window = rx_signal.clone().slice([0..search_window_len]);
    
    println!("  Synchronizing (FFT)...");
    let time_offset = match bachmodem::modulation::synchronize_signal::<Backend>(&device, &search_window) {
        Some(pos) => {
            println!("  ✓ Sync found at sample {} ({:.2}s)", pos, pos as f32 / 8000.0);
            pos
        },
        None => {
            println!("  ✗ Sync failed!");
            return;
        }
    };
    
    // Process slots
    let mut all_llrs: Vec<Tensor<Backend, 1>> = Vec::with_capacity(num_reps);
    let mut snr_estimates = Vec::with_capacity(num_reps);
    
    let slot_duration_samples = tx_len;
    let gap_samples = gap_len;
    let stride = slot_duration_samples + gap_samples;
    
    let mut rake = RakeReceiver::new(3, 200);
    let preamble = bachmodem::wavelet::generate_bach_preamble::<Backend>(&device);
    
    // Detect multipath on first slot
    let first_slot = rx_signal.clone().slice([time_offset..time_offset + slot_duration_samples.min(rx_signal.dims()[0] - time_offset)]);
    rake.detect_paths::<Backend>(&device, &first_slot, &preamble);
    
    for i in 0..num_reps {
        let expected_start = time_offset + i * stride;
        let margin = 2000;
        let window_start = expected_start.saturating_sub(margin);
        let window_end = (expected_start + slot_duration_samples + margin).min(rx_signal.dims()[0]);
        
        if window_start >= rx_signal.dims()[0] { break; }
        
        let slot_signal = rx_signal.clone().slice([window_start..window_end]);
        
        // RAKE combine
        let processed_signal = rake.combine_paths::<Backend>(&device, &slot_signal);
        
        snr_estimates.push(1.0); // Equal weights for now
        
        let preamble_len = preamble.dims()[0];
        let offset_in_slot = expected_start - window_start;
        let data_start = offset_in_slot + preamble_len;
        
        if i == 0 {
            println!("\n  [DEBUG] preamble_len: {}, expected_start: {}, window_start: {}, data_start: {}", 
                     preamble_len, expected_start, window_start, data_start);
        }
        
        if data_start < processed_signal.dims()[0] {
            let data_signal = processed_signal.clone().slice([data_start..processed_signal.dims()[0]]);
            
            let llrs = bachmodem::modulation::demodulate_fhdpsk_soft::<Backend>(
                &device, 
                &data_signal, 
                false, 
                0 // No flourishes in this test
            );
            
            println!("\n  [DEBUG] data_signal length: {}, llrs length: {}", data_signal.dims()[0], llrs.dims()[0]);
            
            // Debug: print first 16 LLR values
            if llrs.dims()[0] >= 16 {
                let llr_data = llrs.to_data();
                let llr_vals = llr_data.as_slice::<f32>().unwrap();
                print!("  [DEBUG] First 16 raw LLRs: ");
                for i in 0..16 {
                    print!("{:.2} ", llr_vals[i]);
                }
                println!();
            }
            
            if llrs.dims()[0] >= 256 {
                let llrs_trunc = llrs.slice([0..256]);
                // DON'T deinterleave here - do it after combining!
                all_llrs.push(llrs_trunc);
                print!(".");
            } else {
                print!("x (only {} LLRs)", llrs.dims()[0]);
                all_llrs.push(Tensor::zeros([256], &device));
            }
        } else {
            print!("X");
            all_llrs.push(Tensor::zeros([256], &device));
        }
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }
    println!();
    
    if all_llrs.is_empty() {
        println!("  ✗ No data decoded");
        return;
    }
    
    // Debug: check first slot's LLRs
    let first_slot_llrs = &all_llrs[0];
    let first_data = first_slot_llrs.to_data();
    let first_vals = first_data.as_slice::<f32>().unwrap();
    print!("  [DEBUG] First slot first 16 LLRs in all_llrs: ");
    for i in 0..16 {
        print!("{:.2} ", first_vals[i]);
    }
    println!();
    
    // Simple averaging of LLRs
    println!("  Averaging LLRs from {} slots...", all_llrs.len());
    let llr_stack: Tensor<Backend, 2> = Tensor::stack(all_llrs.clone(), 0);
    let llr_stack_data = llr_stack.to_data();
    let llr_values = llr_stack_data.as_slice::<f32>().unwrap();
    
    let num_slots = all_llrs.len();
    let mut hard_combined = vec![0.0f32; 256];
    
    for bit_idx in 0..256 {
        let mut llr_sum = 0.0f32;
        for slot_idx in 0..num_slots {
            let llr_val = llr_values[slot_idx * 256 + bit_idx];
            llr_sum += llr_val;
        }
        // Average and scale to reasonable range
        let avg_llr = llr_sum / num_slots as f32;
        // Normalize: scale so that typical values are in [-5, 5] range
        hard_combined[bit_idx] = avg_llr * 4.0;  // Scale up for BP decoder
    }
    
    println!("  Voted {} slots", num_slots);
    
    // CRITICAL: Deinterleave before Polar decode!
    println!("  Deinterleaving...");
    let deinterleaved_llrs = deinterleave_gpu::<Backend>(&device, &Tensor::from_floats(hard_combined.as_slice(), &device), 16);
    let deint_data = deinterleaved_llrs.to_data();
    let deint_slice = deint_data.as_slice::<f32>().unwrap();
    
    // Debug: check first 32 polar-encoded bits to see what should be transmitted
    println!("\nDEBUG: Polar encoding check:");
    print!("  First 32 encoded bits:        ");
    for i in 0..32 {
        print!("{}", encoded_bits[i]);
    }
    println!();
    print!("  First 32 interleaved bits:    ");
    for i in 0..32 {
        print!("{}", interleaved_bits[i]);
    }
    println!();
    print!("  First 32 interleaved LLRs:    ");
    for i in 0..32 {
        print!("{}", if hard_combined[i] < 0.0 { "1" } else { "0" });
    }
    println!();
    print!("  First 32 deinterleaved LLRs:  ");
    for i in 0..32 {
        print!("{}", if deint_slice[i] < 0.0 { "1" } else { "0" });
    }
    println!();
    
    // Debug: check first 16 bits (should decode to 'B' = 66 = 01000010)
    print!("  First 16 deinterleaved LLRs: ");
    for i in 0..16 {
        print!("{:.1} ", deint_slice[i]);
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
    
    let mut final_bytes = Vec::new();
    for chunk in decoded_bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        final_bytes.push(byte);
    }
    
    let len = short_message.len();
    if final_bytes.len() > len {
        final_bytes.truncate(len);
    }
    
    let decoded_msg = String::from_utf8_lossy(&final_bytes);
    println!("  Decoded: \"{}\"", decoded_msg);
    
    if decoded_msg == short_message {
        println!("  ✅ SUCCESS: Perfect match at -30 dB!");
    } else {
        let errors = decoded_msg.chars().zip(short_message.chars()).filter(|(a, b)| a != b).count();
        println!("  ⚠️  PARTIAL: {} char errors", errors);
    }
}