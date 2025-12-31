use burn::tensor::{Tensor, backend::Backend, ElementConversion};
use burn::backend::Wgpu;
use bachmodem::{
    modulate_fhdpsk_with_flourishes, 
    write_wav, 
    watterson::WattersonChannel,
    repetition::{TimeSlotConfig, generate_repetition_transmission},
    wavelet::FS
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("BachModem: Generating Clean WAV with Repetitions and Noise");
    
    let device = Default::default();
    
    // 1. Configuration
    let message = b"BachModem Test Sequence";
    let num_repetitions = 5;
    let listening_gap = 5.0; // 5 seconds gap
    let snr_db = 5.0; // +5 dB SNR (very clean)
    
    println!("Configuration:");
    println!("  Message: {:?}", String::from_utf8_lossy(message));
    println!("  Repetitions: {}", num_repetitions);
    println!("  Gap: {:.1}s", listening_gap);
    println!("  SNR: {:.1} dB", snr_db);
    
    // 2. Create Time Slot Config
    let config = TimeSlotConfig::new(message.len(), num_repetitions, listening_gap);
    
    // 3. Generate Signal with Repetitions
    println!("Generating signal...");
    let signal = generate_repetition_transmission::<Wgpu>(&device, message, &config);
    
    // 4. Add Noise and Channel Effects
    println!("Applying channel effects...");
    
    // Create noise tensor
    let signal_len = signal.dims()[0];
    let noise = Tensor::<Wgpu, 1>::random(
        [signal_len], 
        burn::tensor::Distribution::Normal(0.0, 1.0), 
        &device
    );
    
    // Calculate signal power
    let signal_power = signal.clone().powf_scalar(2.0).mean().into_scalar().elem::<f32>();
    
    // Calculate noise scale for desired SNR
    // SNR_dB = 10 * log10(P_signal / P_noise)
    // P_noise = P_signal / 10^(SNR_dB/10)
    // Noise amplitude scale = sqrt(P_noise)
    let target_noise_power = signal_power / 10.0f32.powf(snr_db / 10.0);
    let noise_scale = target_noise_power.sqrt();
    
    println!("  Signal Power: {:.6}", signal_power);
    println!("  Noise Scale: {:.6}", noise_scale);
    
    // Apply Watterson Channel (Mild)
    let channel = WattersonChannel::gentle();
    let faded_signal = channel.apply::<Wgpu>(&device, &signal);
    
    // Add noise
    let noisy_signal = faded_signal + (noise * noise_scale);
    
    // 5. Add Initial Noise Period (Random Start)
    // Add 2-5 seconds of pure noise at the beginning
    let start_delay_sec = 3.0; // Fixed for reproducibility, or use rand
    let start_delay_samples = (start_delay_sec * FS) as usize;
    
    let initial_noise = Tensor::<Wgpu, 1>::random(
        [start_delay_samples], 
        burn::tensor::Distribution::Normal(0.0, 1.0), 
        &device
    ) * noise_scale;
    
    let final_signal = Tensor::cat(vec![initial_noise, noisy_signal], 0);
    
    // 6. Write to WAV
    let output_path = "bach_clean_5reps.wav";
    println!("Writing to {}...", output_path);
    write_wav(&final_signal, output_path)?;
    
    println!("Done! Generated {:.1}s audio file.", final_signal.dims()[0] as f32 / (FS as f32));
    
    Ok(())
}
