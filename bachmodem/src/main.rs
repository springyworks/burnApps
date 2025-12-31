use bachmodem::*;
use burn::backend::Wgpu;
use bachmodem::write_wav;

type MyBackend = Wgpu;

fn main() {
    println!("=======================================================");
    println!("   BachModem - Musical Wavelet Modem for HF Radio");
    println!("=======================================================");
    println!();
    println!("Physical Layer:");
    println!("  - Sampling Rate: 8000 Hz");
    println!("  - Symbol Duration: 2.0 seconds");
    println!("  - Carrier Frequencies: C4 (261.63 Hz) to D6 (1174.66 Hz)");
    println!("  - Modulation: FH-DPSK (Frequency-Hopping DPSK)");
    println!("  - Bandwidth: 200 Hz - 2.8 kHz (SSB compatible)");
    println!();
    
    // Initialize device
    let device = Default::default();
    
    // Message to transmit
    let message = "Hello from BachModem! This is a deep-space wavelet transmission using Bach themes. The music you hear encodes digital data using Morlet wavelets mapped to the C-Major scale. Each symbol is 2 seconds long, allowing detection at -30 dB SNR over HF channels. 73 de AI";
    
    println!("Message to transmit:");
    println!("  \"{}\"", message);
    println!();
    println!("Encoding {} bytes into wavelet sequence...", message.len());
    
    // Modulate the message with musical flourishes
    let data_bytes = message.as_bytes();
    println!("Generating FH-DPSK signal with Bach Preamble and musical flourishes...");
    
    let signal = modulation::modulate_fhdpsk_with_flourishes::<MyBackend>(
        &device,
        data_bytes,
        true, // Add preamble
        128,  // Add flourish every 128 symbols (every ~4 minutes)
    );
    
    let signal_len = signal.dims()[0];
    let duration_seconds = signal_len as f64 / wavelet::FS;
    
    println!("Signal generated:");
    println!("  - Length: {} samples", signal_len);
    println!("  - Duration: {:.2} seconds", duration_seconds);
    println!();
    
    // Calculate transmission details
    let bits = data_bytes.len() * 8;
    let padded_bits = ((bits + 15) / 16) * 16; // Padded to multiple of 16
    let total_bits = padded_bits + 16; // Include reference block
    let num_symbols = total_bits;
    let data_duration = num_symbols as f64 * wavelet::SYMBOL_DURATION;
    let preamble_duration = duration_seconds - data_duration;
    
    println!("Transmission structure:");
    println!("  - Preamble: {:.1} seconds (Bach Sweep for sync)", preamble_duration);
    println!("  - Data: {:.1} seconds ({} symbols)", data_duration, num_symbols);
    println!("  - Bit rate: {:.2} bits/second", bits as f64 / data_duration);
    println!();
    
    // Save to WAV file
    let output_path = "bachmodem_output.wav";
    println!("Writing WAV file to '{}'...", output_path);
    
    match write_wav(&signal, output_path) {
        Ok(_) => {
            println!("✓ WAV file written successfully!");
            println!();
            println!("Listen to the beautiful wavelet music!");
            println!("Each frequency sweep encodes data bits differentially.");
            println!();
            println!("Frequency mapping (C-Major scale):");
            for (i, freq) in wavelet::BACH_FREQUENCIES.iter().enumerate() {
                let note_names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
                                  "D5", "E5", "F5", "G5", "A5", "B5", "C6", "D6"];
                println!("  {:2X}: {} = {:.2} Hz", i, note_names[i], freq);
            }
            println!();
            println!("Hopping pattern (melodic intervals):");
            print!("  ");
            for (i, &idx) in wavelet::HOPPING_PATTERN.iter().enumerate() {
                print!("{:X}", idx);
                if i < wavelet::HOPPING_PATTERN.len() - 1 {
                    print!(" -> ");
                }
                if (i + 1) % 8 == 0 && i < wavelet::HOPPING_PATTERN.len() - 1 {
                    print!("\n  ");
                }
            }
            println!();
            println!();
            println!("=======================================================");
            println!("Transmission complete! 73 de BachModem");
            println!("=======================================================");
        }
        Err(e) => {
            eprintln!("Error writing WAV file: {}", e);
        }
    }
}
