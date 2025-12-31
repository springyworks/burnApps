use burn::tensor::{Tensor, backend::Backend};
use hound::{WavWriter, WavSpec};
use std::path::Path;

/// WAV file parameters for BachModem
pub const WAV_SAMPLE_RATE: u32 = 8000;
pub const WAV_BITS_PER_SAMPLE: u16 = 16;
pub const WAV_CHANNELS: u16 = 1; // Mono

/// Normalize and scale signal on GPU (stays on GPU!)
/// 
/// **NO SYNC POINT**: This is pure GPU operation for preparing audio
pub fn prepare_wav_signal_gpu<B: Backend>(
    signal: &Tensor<B, 1>,
) -> Tensor<B, 1> {
    // Find max amplitude for normalization (stays on GPU)
    let abs_signal = signal.clone().abs();
    let max_amp = abs_signal.max();
    
    // Normalize to [-1, 1] range
    let normalized = signal.clone() / max_amp.clone().clamp_min(1e-8);
    
    // Scale to 16-bit PCM range
    normalized * 32767.0
}

/// Writes a Burn tensor to a WAV file
/// ⚠️ **SYNC POINT**: This downloads tensor to CPU for file I/O
/// Writes a Burn tensor to a WAV file
/// ⚠️ **SYNC POINT**: This downloads tensor to CPU for file I/O
/// 
/// Use prepare_wav_signal_gpu() first to keep normalization on GPU
/// The signal is normalized to the range [-1.0, 1.0] and then scaled to 16-bit PCM
pub fn write_wav<B: Backend, P: AsRef<Path>>(
    signal: &Tensor<B, 1>,
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    // ⚠️ SYNC POINT: Convert tensor to Vec<f32>
    let data = signal.clone().into_data();
    let samples: Vec<f32> = data.to_vec::<f32>().unwrap();
    
    // Find max amplitude for normalization
    let max_amp = samples.iter()
        .map(|&x| x.abs())
        .fold(0.0f32, f32::max);
    
    let scale = if max_amp > 0.0 {
        1.0 / max_amp
    } else {
        1.0
    };
    
    // Create WAV file
    let spec = WavSpec {
        channels: WAV_CHANNELS,
        sample_rate: WAV_SAMPLE_RATE,
        bits_per_sample: WAV_BITS_PER_SAMPLE,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = WavWriter::create(path, spec)?;
    
    // Write samples as 16-bit PCM
    for sample in samples {
        let normalized = sample * scale;
        let pcm_value = (normalized * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(pcm_value)?;
    }
    
    writer.finalize()?;
    Ok(())
}

/// Reads a WAV file into a Burn tensor
pub fn read_wav<B: Backend>(
    device: &B::Device,
    path: &Path,
) -> Result<Tensor<B, 1>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    
    // Read samples and convert to f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            reader.samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .map(|s| s.unwrap())
                .collect()
        }
    };
    
    Ok(Tensor::from_floats(samples.as_slice(), device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use std::f32::consts::PI;
    
    type TestBackend = Wgpu;
    
    #[test]
    fn test_write_wav() {
        let device = Default::default();
        
        // Generate a simple sine wave at 440 Hz
        let duration = 1.0; // seconds
        let freq = 440.0;
        let sample_rate = WAV_SAMPLE_RATE as f32;
        let num_samples = (duration * sample_rate) as usize;
        
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * PI * freq * t).sin()
            })
            .collect();
        
        let signal = Tensor::<TestBackend, 1>::from_floats(samples.as_slice(), &device);
        
        // Write to file
        let path = "test_output.wav";
        write_wav(&signal, path).expect("Failed to write WAV file");
        
        // Clean up
        std::fs::remove_file(path).ok();
        
        println!("WAV file test successful");
    }
}
