use bachmodem::{modulate_fhdpsk_with_flourishes, demodulate_fhdpsk_ex, write_wav, read_wav};
use burn::backend::Wgpu;

type Backend = Wgpu;

fn main() {
    println!("=======================================================");
    println!("   BachModem Decoder Test");
    println!("=======================================================\n");
    
    let device = Default::default();
    
    // Original message
    let original_message = "Hello, Bach! This message has musical flourishes throughout the transmission. Each flourish is a rapid arpeggio sweep that sounds beautiful while helping synchronization. 73!";
    println!("Original message:");
    println!("  \"{}\"\n", original_message);
    println!("Length: {} bytes\n", original_message.len());
    
    // Modulate with flourishes every 64 symbols
    println!("Modulating with musical flourishes every 64 symbols...");
    let signal = modulate_fhdpsk_with_flourishes::<Backend>(
        &device,
        original_message.as_bytes(),
        true,  // Add preamble
        64,    // Insert flourish every 64 symbols (musically pleasing)
    );
    
    let signal_len = signal.dims()[0];
    let duration = signal_len as f64 / 8000.0;
    println!("Signal generated: {} samples, {:.1} seconds\n", signal_len, duration);
    
    // Save to WAV
    let wav_path = "test_decoder.wav";
    println!("Writing to {}...", wav_path);
    write_wav(&signal, wav_path).expect("Failed to write WAV");
    println!("✓ WAV written\n");
    
    // Now decode it back!
    println!("=======================================================");
    println!("   DECODER TEST - Reading back the signal");
    println!("=======================================================\n");
    
    println!("Reading WAV file...");
    let received_signal = read_wav::<Backend>(&device, wav_path.as_ref())
        .expect("Failed to read WAV");
    println!("✓ Signal loaded\n");
    
    println!("Demodulating signal...");
    let decoded_bytes = demodulate_fhdpsk_ex::<Backend>(
        &device,
        &received_signal,
        true, // Use synchronization
        64,   // Same flourish interval as encoding
    );
    
    if decoded_bytes.is_empty() {
        println!("\n❌ Decoding failed - no data recovered\n");
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
            
            // Compare
            let original_bytes = original_message.as_bytes();
            let match_len = decoded_bytes.len().min(original_bytes.len());
            let mut errors = 0;
            
            for i in 0..match_len {
                if decoded_bytes[i] != original_bytes[i] {
                    errors += 1;
                }
            }
            
            let ber = if match_len > 0 {
                (errors as f64 / (match_len * 8) as f64) * 100.0
            } else {
                100.0
            };
            
            println!("Statistics:");
            println!("  Original: {} bytes", original_bytes.len());
            println!("  Decoded:  {} bytes", decoded_bytes.len());
            println!("  Errors:   {} bytes", errors);
            println!("  BER:      {:.2}%", ber);
            
            if errors == 0 && decoded_bytes.len() >= original_bytes.len() {
                println!("\n✨ PERFECT DECODING! ✨\n");
            } else if errors < 5 {
                println!("\n✓ Good decoding (minor errors)\n");
            } else {
                println!("\n⚠ Decoding has errors\n");
            }
        }
        Err(_) => {
            println!("\n⚠ Decoded bytes are not valid UTF-8");
            println!("Decoded {} bytes (hex): {:02X?}\n", decoded_bytes.len(), &decoded_bytes[..decoded_bytes.len().min(64)]);
        }
    }
    
    println!("=======================================================");
    println!("Test complete!");
    println!("=======================================================");
}
