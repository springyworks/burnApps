/// Time-Slotted Repetition Test (-30 dB SNR)
/// 
/// Demonstrates:
/// 1. High repetition count (15x) for extreme weak signal recovery
/// 2. Time-slotted transmission with listening gaps (TDD)
/// 3. Precise timing alignment using clock
/// 4. Maximum Ratio Combining (MRC) of all slots

use bachmodem::{
    WattersonChannel,
    interleave, deinterleave, 
    PolarCode, soft_bits_to_llrs, compute_soft_bits,
    TimeSlotConfig, generate_repetition_transmission,
    soft_combine_gpu, estimate_snr_from_correlation,
    RakeReceiver,
    modulate_fhdpsk_with_flourishes,
    deinterleave_gpu,
    FftBackend,
};
use burn::backend::wgpu::{CubeBackend, WgpuRuntime, WgpuDevice};
use burn::tensor::{Tensor, Distribution, ElementConversion};
use hound;

// Use raw CubeBackend to avoid Fusion wrapper which doesn't implement FftBackend yet
type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Time-Slotted Repetition Test (-30 dB SNR)              ║");
    println!("║  15 Repetitions with Listening Gaps                     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    let device = Default::default();
    let snr_db = -30.0;
    let filename = "time_slot_test.wav";
    
    // ========================================================================
    // PART 1: GENERATE SIGNAL WITH TIME SLOTS
    // ========================================================================
    println!("Generating signal for {} dB SNR...", snr_db);
    
    let message = "BachModem!";
    let num_reps = 15; // Increased repetitions for -30 dB
    let listening_gap = 5.0; // 5 seconds gap for listening
    let list_size = 8;
    
    // 1. Encode
    let data_bytes = message.as_bytes();
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
    
    let mut tx_bytes = Vec::new();
    for chunk in interleaved_bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        tx_bytes.push(byte);
    }
    
    // 2. Generate repetitions with gaps
    println!("Configuring Time Slots:");
    let config = TimeSlotConfig::new(tx_bytes.len(), num_reps, listening_gap);
    println!("  • Transmission duration: {:.2}s", config.transmission_duration);
    println!("  • Listening gap: {:.2}s", config.listening_gap);
    println!("  • Total cycle time: {:.2}s", config.transmission_duration + config.listening_gap);
    println!("  • Total duration: {:.2}s", config.total_duration());
    
    let clean_signal = generate_repetition_transmission::<Backend>(&device, &tx_bytes, &config);
    
    // 3. Channel
    let signal_power: f32 = clean_signal.clone().powf_scalar(2.0).mean().into_scalar().elem();
    let snr_linear = 10f32.powf(snr_db / 10.0);
    let noise_std = (signal_power / snr_linear).sqrt();
    
    println!("  Signal power: {:.6}, Noise std: {:.6}", signal_power, noise_std);
    
    let channel = WattersonChannel::moderate();
    let faded_signal = channel.apply::<Backend>(&device, &clean_signal);
    
    let noise = Tensor::<Backend, 1>::random(
        faded_signal.shape(),
        Distribution::Normal(0.0, noise_std as f64),
        &device,
    );
    
    let rx_signal = faded_signal + noise;
    
    // 4. Save to WAV
    println!("Saving to {}...", filename);
    let rx_data = rx_signal.to_data();
    let rx_samples = rx_data.as_slice::<f32>().unwrap();
    
    // Normalize to prevent clipping
    let max_val = rx_samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!("  Max sample value: {:.6} (will be normalized)", max_val);
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
    println!("  ✓ Saved {} samples to {}", rx_samples.len(), filename);
    
    // Free tensors
    drop(rx_signal);
    drop(clean_signal);
    
    // ========================================================================
    // PART 2: READ AND DECODE WAV
    // ========================================================================
    println!("\nReading and decoding {}...", filename);
    
    let mut reader = hound::WavReader::open(filename).unwrap();
    let samples: Vec<f32> = reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32767.0)
        .collect();
        
    println!("  Loaded {} samples", samples.len());
    
    let rx_signal = Tensor::<Backend, 1>::from_floats(samples.as_slice(), &device);
    
    // 4. Receiver: Find first repetition
    let search_window_len = 100000.min(rx_signal.dims()[0]);
    let search_window = rx_signal.clone().slice([0..search_window_len]);
    
    println!("  Synchronizing...");
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
    
    // 5. Process each repetition
    let mut all_llrs: Vec<Tensor<Backend, 1>> = Vec::with_capacity(num_reps);
    let mut snr_estimates = Vec::with_capacity(num_reps);
    
    let slot_duration_samples = (config.transmission_duration * 8000.0) as usize;
    let gap_samples = (config.listening_gap * 8000.0) as usize;
    let stride = slot_duration_samples + gap_samples;
    
    // Detect multipath
    let mut rake = RakeReceiver::new(3, 200);
    let preamble = bachmodem::wavelet::generate_bach_preamble::<Backend>(&device);
    println!("  Detecting multipath structure...");
    let first_slot = rx_signal.clone().slice([time_offset..time_offset + slot_duration_samples.min(rx_signal.dims()[0] - time_offset)]);
    rake.detect_paths::<Backend>(&device, &first_slot, &preamble);
    
    for i in 0..num_reps {
        let expected_start = time_offset + i * stride;
        let margin = 2000;
        let window_start = expected_start.saturating_sub(margin);
        let window_end = (expected_start + slot_duration_samples + margin).min(rx_signal.dims()[0]);
        
        if window_start >= rx_signal.dims()[0] { break; }
        
        let slot_signal = rx_signal.clone().slice([window_start..window_end]);
        
        // RAKE combining
        let processed_signal = rake.combine_paths::<Backend>(&device, &slot_signal);
        
        snr_estimates.push(10.0); // Increased confidence for MRC
        
        // Skip preamble manually to avoid second sync failure
        let preamble_len = preamble.dims()[0];
        let offset_in_slot = expected_start - window_start;
        let data_start = offset_in_slot + preamble_len;
        
        let data_signal = if data_start < processed_signal.dims()[0] {
            processed_signal.clone().slice([data_start..processed_signal.dims()[0]])
        } else {
            processed_signal.clone() // Should not happen
        };

        // Demodulate without internal sync
        let llrs = bachmodem::modulation::demodulate_fhdpsk_soft::<Backend>(
            &device, 
            &data_signal, 
            false, // Disable internal sync
            64
        );
        
        drop(processed_signal);
        drop(data_signal);
        
        let llrs_len = llrs.dims()[0];
        
        if llrs_len >= 256 {
            let llrs_trunc = llrs.slice([0..256]);
            let deint_llrs_tensor = deinterleave_gpu::<Backend>(&device, &llrs_trunc, 16);
            all_llrs.push(deint_llrs_tensor);
            println!("    Rep {}/{}: Decoded {} LLRs", i+1, num_reps, llrs_len);
            drop(slot_signal);
        } else {
            println!("    Rep {}/{}: Failed (got {} bits)", i+1, num_reps, llrs_len);
            all_llrs.push(Tensor::zeros([256], &device));
        }
    }
    
    if all_llrs.is_empty() {
        println!("  ✗ No data decoded");
        return;
    }
    
    // 6. Soft combining
    println!("  Combining {} repetitions...", all_llrs.len());
    let llr_stack = Tensor::stack(all_llrs, 0);
    let weights = Tensor::from_floats(snr_estimates.as_slice(), &device);
    let combined_llrs = soft_combine_gpu(&llr_stack, &weights);
    
    // 7. Decode
    let combined_data = combined_llrs.to_data();
    let combined_slice = combined_data.as_slice::<f32>().unwrap();
    let llr_vec: Vec<f32> = combined_slice.to_vec();
    
    println!("  Decoding with Polar SCL (L={})...", list_size);
    let decoded_bits = polar.decode_scl(&llr_vec, list_size);
    
    // 8. Convert to bytes
    let mut final_bytes = Vec::new();
    for chunk in decoded_bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        final_bytes.push(byte);
    }
    
    let len = message.len();
    if final_bytes.len() > len {
        final_bytes.truncate(len);
    }
    
    println!("  Decoded bytes: {:02X?}", final_bytes);
    let decoded_msg = String::from_utf8_lossy(&final_bytes);
    println!("  Decoded string: \"{}\"", decoded_msg);
    
    let errors = decoded_msg.chars()
        .zip(message.chars())
        .filter(|(a, b)| a != b)
        .count();
    
    if decoded_msg == message {
        println!("  ✅ SUCCESS: Perfect match!");
    } else {
        println!("  ⚠️  PARTIAL: {} char errors", errors);
    }
}
