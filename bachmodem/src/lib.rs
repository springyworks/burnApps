//! BachModem - Musical Wavelet Modem for HF Radio
//! 
//! A data modem for HF radio weak signal communication that uses Bach themes.
//! Implements Frequency-Hopping Differential Phase Shift Keying (FH-DPSK)
//! with Morlet wavelets mapped to a C-Major scale.
//! 
//! Based on WaveletsJAX Python/JAX implementation, achieving -30 dB SNR
//! communication over HF-Watterson channels.

pub mod wavelet;
pub mod modulation;
pub mod wav;
pub mod watterson;
pub mod repetition;
pub mod interleaver;
pub mod polar;
pub mod polar_bp;
pub mod rake;
pub mod gpu_ops;
pub mod deinterleave_gpu;
pub mod gpu_test_utils;
pub mod gpu_math;
pub mod fft_correlation;

pub use wavelet::{BACH_FREQUENCIES, HOPPING_PATTERN, FS, SYMBOL_DURATION, generate_bach_flourish};
pub use modulation::{modulate_fhdpsk, modulate_fhdpsk_with_flourishes, demodulate_fhdpsk, demodulate_fhdpsk_ex, demodulate_fhdpsk_soft, synchronize_signal, synchronize_signal_gpu, encode_bits, pack_bits};
pub use wav::{write_wav, read_wav, prepare_wav_signal_gpu};
pub use watterson::WattersonChannel;
pub use repetition::{TimeSlotConfig, generate_repetition_transmission, CombiningStrategy, DecodedCopy, combine_decoded_copies};
pub use interleaver::{interleave, deinterleave};
pub use polar::{PolarCode, soft_bits_to_llrs, compute_soft_bits, crc8, encode_with_crc, verify_crc};
pub use polar_bp::PolarCodeBP;
pub use rake::{RakeReceiver, RakeFinger, estimate_rake_gain};
pub use gpu_ops::{cross_correlation_gpu, soft_combine_gpu, coherent_combine_symbols, estimate_snr_from_correlation, estimate_snr_from_correlation_gpu};
pub use deinterleave_gpu::{deinterleave_gpu, interleave_gpu};
pub use gpu_test_utils::{assert_approx_eq_gpu, assert_approx_eq_scalar, validate_roundtrip, assert_normalized};
pub use gpu_math::{atan2_fast_gpu};
pub use fft_correlation::{fft_cross_correlation, cross_correlation_fft, FftBackend};
