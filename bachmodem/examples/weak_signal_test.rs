use bachmodem::{modulate_fhdpsk_with_flourishes, demodulate_fhdpsk_ex, write_wav, read_wav};
use burn::backend::Wgpu;
use burn::tensor::{Tensor, Distribution};
use rand::Rng;

type Backend = Wgpu;

fn main() {
    println!("=======================================================");
    println!("   BachModem Weak Signal Test (-20 dB SNR)");
    println!("=======================================================\n");
    
    let device = Default::default();
    let mut rng = rand::thread_rng();
    
    // Original message
    let original_message = "Test msg 73!";
    println!("Original message:");
    println!("  \"{}\"\n", original_message);
    println!("Length: {} bytes\n", original_message.len());
    
    // Generate clean signal with flourishes every 64 symbols
    println!("Generating clean BachModem signal...");
    let clean_signal = modulate_fhdpsk_with_flourishes::<Backend>(
        &device,
        original_message.as_bytes(),
        true,  // Add preamble
        64,    // Flourish interval
    );
    
    let signal_len = clean_signal.dims()[0];
    let duration = signal_len as f64 / 8000.0;
    println!("Clean signal: {} samples, {:.1} seconds\n", signal_len, duration);
    
    // Calculate signal power
    let signal_power = clean_signal.clone().powf_scalar(2.0).mean().into_scalar();
    println!("Signal power: {:.6}", signal_power);
    
    // Target SNR: -20 dB
    let target_snr_db = -20.0;
    let target_snr_linear = 10f32.powf(target_snr_db / 10.0);
    
    // Calculate required noise power: SNR = P_signal / P_noise
    let noise_power = signal_power / target_snr_linear;
    let noise_std = noise_power.sqrt();
    
    println!("Target SNR: {} dB ({:.6} linear)", target_snr_db, target_snr_linear);
    println!("Noise power: {:.6}", noise_power);
    println!("Noise std dev: {:.6}\n", noise_std);
    
    // Random offset: place signal somewhere between 5-15 seconds into the noise
    let min_offset_seconds = 5.0;
    let max_offset_seconds = 15.0;
    let offset_seconds = rng.gen_range(min_offset_seconds..max_offset_seconds);
    let offset_samples = (offset_seconds * 8000.0) as usize;
    
    println!("Random timing offset: {:.2} seconds ({} samples)\n", offset_seconds, offset_samples);
    
    // Create noisy channel: [noise] + [signal] + [noise]
    let total_len = offset_samples + signal_len + 80000; // Extra 10 seconds of noise after signal
    println!("Generating noisy channel ({} samples, {:.1} seconds)...", total_len, total_len as f64 / 8000.0);
    
    // Generate white Gaussian noise
    let noise = Tensor::<Backend, 1>::random(
        [total_len],
        Distribution::Normal(0.0, noise_std as f64),
        &device,
    );
    
    // Insert clean signal at random offset
    println!("Inserting signal at offset {} samples...", offset_samples);
    let mut noisy_signal_data = noise.to_data();
    let clean_signal_data = clean_signal.to_data();
    
    // Copy signal into noise buffer
    for i in 0..signal_len {
        let idx = offset_samples + i;
        if idx < total_len {
            let noise_val: f32 = noisy_signal_data.as_slice::<f32>().unwrap()[idx];
            let signal_val: f32 = clean_signal_data.as_slice::<f32>().unwrap()[i];
            // Add signal to noise (not replace - this is how real radio works)
            noisy_signal_data.as_mut_slice::<f32>().unwrap()[idx] = noise_val + signal_val;
        }
    }
    
    let noisy_signal = Tensor::<Backend, 1>::from_data(noisy_signal_data, &device);
    
    // Verify actual SNR achieved
    let actual_noise_power = noise.clone().powf_scalar(2.0).mean().into_scalar();
    let actual_snr_linear = signal_power / actual_noise_power;
    let actual_snr_db = 10.0 * actual_snr_linear.log10();
    
    println!("Actual noise power: {:.6}", actual_noise_power);
    println!("Actual SNR: {:.2} dB\n", actual_snr_db);
    
    // Write noisy signal to WAV
    let wav_path = "weak_signal_test.wav";
    println!("Writing noisy signal to {}...", wav_path);
    write_wav(&noisy_signal, wav_path).expect("Failed to write WAV");
    println!("✓ WAV written\n");
    
    // Now attempt to decode from the noisy channel
    println!("=======================================================");
    println!("   DECODER TEST - Recovering signal from noise");
    println!("=======================================================\n");
    
    println!("Reading noisy WAV file...");
    let received_signal = read_wav::<Backend>(&device, wav_path.as_ref())
        .expect("Failed to read WAV");
    println!("✓ Signal loaded ({} samples)\n", received_signal.dims()[0]);
    
    println!("Searching for preamble in {} seconds of noisy data...", received_signal.dims()[0] as f64 / 8000.0);
    println!("Demodulating signal...");
    let decoded_bytes = demodulate_fhdpsk_ex::<Backend>(
        &device,
        &received_signal,
        true,  // Use synchronization (critical for finding signal in noise!)
        64,    // Same flourish interval as encoding
    );
    
    if decoded_bytes.is_empty() {
        println!("\n❌ DECODING FAILED - No data recovered");
        println!("   Signal may be below detection threshold at {} dB SNR\n", target_snr_db);
        return;
    }
    
    // Try to decode as UTF-8
    match String::from_utf8(decoded_bytes.clone()) {
        Ok(decoded_message) => {
            println!("\n=======================================================");
            println!("   DECODING SUCCESS!");
            println!("=======================================================\n");
            println!("Decoded message:");
            println!("  \"{}\"\n", decoded_message);
            
            // Compare with original
            let original_bytes = original_message.as_bytes();
            let match_len = decoded_bytes.len().min(original_bytes.len());
            
            let mut bit_errors = 0;
            for i in 0..match_len {
                let diff = decoded_bytes[i] ^ original_bytes[i];
                bit_errors += diff.count_ones() as usize;
            }
            
            let total_bits = match_len * 8;
            let ber = if total_bits > 0 {
                (bit_errors as f64 / total_bits as f64) * 100.0
            } else {
                100.0
            };
            
            println!("Statistics:");
            println!("  Channel SNR:    {} dB", target_snr_db);
            println!("  Timing offset:  {:.2} seconds (random)", offset_seconds);
            println!("  Original:       {} bytes", original_bytes.len());
            println!("  Decoded:        {} bytes", decoded_bytes.len());
            println!("  Bit errors:     {} bits", bit_errors);
            println!("  Total bits:     {} bits", total_bits);
            println!("  BER:            {:.4}%", ber);
            
            if bit_errors == 0 && decoded_bytes.len() >= original_bytes.len() {
                println!("\n✨ PERFECT DECODING AT {} dB SNR! ✨", target_snr_db);
                println!("   BachModem successfully recovered signal from noisy channel!\n");
            } else if ber < 1.0 {
                println!("\n✓ EXCELLENT DECODING (BER < 1%)\n");
            } else if ber < 5.0 {
                println!("\n✓ Good decoding (BER < 5%)\n");
            } else if ber < 10.0 {
                println!("\n⚠ Fair decoding (BER < 10%) - may need FEC\n");
            } else {
                println!("\n⚠ Poor decoding (BER >= 10%) - FEC required\n");
            }
        }
        Err(_) => {
            println!("\n⚠ Decoded bytes are not valid UTF-8");
            println!("Decoded {} bytes (hex): {:02X?}\n", decoded_bytes.len(), &decoded_bytes[..decoded_bytes.len().min(64)]);
            println!("This may indicate severe channel errors at {} dB SNR\n", target_snr_db);
        }
    }
    
    println!("=======================================================");
    println!("Weak signal test complete!");
    println!("=======================================================");
    println!("\nKey findings:");
    println!("  - Preamble sync: {}", if decoded_bytes.is_empty() { "FAILED" } else { "PASSED" });
    println!("  - Signal detection: {}", if decoded_bytes.is_empty() { "FAILED" } else { "PASSED" });
    println!("  - Data recovery: {}", if decoded_bytes.is_empty() { "FAILED" } else { "PARTIAL/COMPLETE" });
    println!("\nNext steps:");
    println!("  - Try -25 dB, -30 dB SNR for deeper tests");
    println!("  - Add frequency-selective fading (Watterson model)");
    println!("  - Implement Polar code FEC for error correction");
}
