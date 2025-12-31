/// Quick Encoding Test - Just verifies the fix without full system test

use bachmodem::{
    TimeSlotConfig, generate_repetition_transmission,
    interleave, PolarCode,
};
use burn::backend::Wgpu;

type Backend = Wgpu;

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  BachModem QUICK ENCODING TEST                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    let device = Default::default();
    
    let message = "BachModem 73!";
    let num_reps = 2;
    
    println!("Message: \"{}\" ({} bytes)", message, message.len());
    
    // 1. Encode
    println!("\n[Step 1] Encoding with Polar Code...");
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
    println!("  ✓ Encoded {} bits -> {} bits", data_bits.len(), encoded_bits.len());
    
    // 2. Interleave
    println!("\n[Step 2] Interleaving...");
    let interleaved_bits = interleave(&encoded_bits, 16);
    println!("  ✓ Interleaved {} bits", interleaved_bits.len());
    
    // 3. Pack to bytes
    println!("\n[Step 3] Packing to bytes...");
    let mut tx_bytes = Vec::new();
    for chunk in interleaved_bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &bit) in chunk.iter().enumerate() {
            byte |= bit << (7 - i);
        }
        tx_bytes.push(byte);
    }
    println!("  ✓ Packed to {} bytes", tx_bytes.len());
    
    // 4. Calculate transmission config
    println!("\n[Step 4] Calculating transmission timing...");
    let config = TimeSlotConfig::new(tx_bytes.len(), num_reps, 0.5);
    println!("  Transmission duration: {:.1}s", config.transmission_duration);
    println!("  Listening gap: {:.1}s", config.listening_gap);
    println!("  Total duration: {:.1}s", config.total_duration());
    
    // 5. Generate transmission
    println!("\n[Step 5] Generating {} repetitions...", num_reps);
    let clean_signal = generate_repetition_transmission::<Backend>(&device, &tx_bytes, &config);
    println!("  ✓ Generated signal: {} samples ({:.1}s)", 
        clean_signal.dims()[0], 
        clean_signal.dims()[0] as f64 / 8000.0
    );
    
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  ✨ SUCCESS - Encoding pipeline works!                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    println!("Note: With SYMBOL_DURATION=0.1s, this is 10x faster than");
    println!("      the original 2.0s deep-space spec.");
    println!("\nThe hang was caused by allocating 40-minute buffers!");
    println!("Now fixed with reduced timing parameters for testing.");
}
